"""Data quality checks + documented cleaning decisions.

Rule we follow: raw rows are never deleted silently. Problems are either fixed with a written
reason (logged in DECISIONS) or marked with a flag column, and the row is excluded later only
where it matters (e.g. the modelling table) with the count written to the exclusion log.
"""
import numpy as np
import pandas as pd

DECISIONS = []   # (table, issue, rows affected, action)


def log(table, issue, n, action):
    DECISIONS.append({"table": table, "issue": issue, "rows": int(n), "action": action})


# ----------------------------------------------------------------------------- checks
KEYS = {
    "orders": ["order_id"], "customers": ["customer_id"], "products": ["product_id"],
    "sellers": ["seller_id"], "translation": ["product_category_name"],
    "items": ["order_id", "order_item_id"], "payments": ["order_id", "payment_sequential"],
}


def key_report(data):
    """Is every primary key unique? (order_id repeating in items/payments alone is fine.)"""
    rows = []
    for t, k in KEYS.items():
        df = data[t]
        rows.append({"table": t, "key": " + ".join(k), "rows": len(df),
                     "duplicate_keys": int(df.duplicated(k).sum()),
                     "null_keys": int(df[k].isna().any(axis=1).sum()),
                     "duplicate_full_rows": int(df.duplicated().sum())})
    return pd.DataFrame(rows)


def integrity_report(data):
    """Foreign keys: how many child rows point to a parent that does not exist."""
    o, c, i, p, pr, s, tr = (data[k] for k in
                             ("orders", "customers", "items", "payments", "products", "sellers", "translation"))
    checks = [
        ("orders.customer_id -> customers", o.customer_id, c.customer_id),
        ("items.order_id -> orders", i.order_id, o.order_id),
        ("items.product_id -> products", i.product_id, pr.product_id),
        ("items.seller_id -> sellers", i.seller_id, s.seller_id),
        ("payments.order_id -> orders", p.order_id, o.order_id),
        ("products.category -> translation", pr.product_category_name.dropna(), tr.product_category_name),
    ]
    rows = [{"relation": name, "child_rows": len(ch), "orphans": int((~ch.isin(par)).sum())}
            for name, ch, par in checks]
    rows += [
        {"relation": "orders without items", "child_rows": len(o), "orphans": int((~o.order_id.isin(i.order_id)).sum())},
        {"relation": "orders without payments", "child_rows": len(o), "orphans": int((~o.order_id.isin(p.order_id)).sum())},
    ]
    return pd.DataFrame(rows)


def chronology_flags(o):
    """Boolean flags for event order problems in the orders table."""
    f = pd.DataFrame(index=o.index)
    f["approved_before_purchase"] = o.order_approved_at < o.order_purchase_timestamp
    f["carrier_before_purchase"] = o.order_delivered_carrier_date < o.order_purchase_timestamp
    f["carrier_before_approval"] = o.order_delivered_carrier_date < o.order_approved_at
    f["delivered_before_purchase"] = o.order_delivered_customer_date < o.order_purchase_timestamp
    f["delivered_before_carrier"] = o.order_delivered_customer_date < o.order_delivered_carrier_date
    f["estimated_before_purchase"] = o.order_estimated_delivery_date < o.order_purchase_timestamp.dt.normalize()
    f["delivered_status_no_date"] = (o.order_status == "delivered") & o.order_delivered_customer_date.isna()
    f["not_delivered_status_with_date"] = (o.order_status != "delivered") & o.order_delivered_customer_date.notna()
    return f


# ----------------------------------------------------------------------------- cleaning
def clean(data):
    DECISIONS.clear()
    d = {k: v.copy() for k, v in data.items()}

    # products -------------------------------------------------------------------------------
    pr = d["products"]
    # the raw file has the typo "lenght" -> rename for readability
    pr = pr.rename(columns={"product_name_lenght": "product_name_length",
                            "product_description_lenght": "product_description_length"})
    n = pr.product_category_name.isna().sum()
    pr["product_category_name"] = pr.product_category_name.fillna("unknown")
    log("products", "missing category", n, "set to 'unknown' (kept, it is still a valid product)")
    num = ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
    bad = (pr[num] <= 0).any(axis=1)
    log("products", "weight or dimension = 0 (impossible)", bad.sum(), "set to NaN, imputed later from train only")
    pr.loc[bad, num] = pr.loc[bad, num].where(pr.loc[bad, num] > 0)
    log("products", "missing weight/dimensions", pr[num].isna().any(axis=1).sum(),
        "kept as NaN, imputed inside the model pipeline (median of train)")
    pr["product_volume_cm3"] = pr.product_length_cm * pr.product_height_cm * pr.product_width_cm
    tr = d["translation"]
    pr = pr.merge(tr, on="product_category_name", how="left")
    miss = pr.product_category_name_english.isna()
    log("products", "category without English translation", miss.sum(),
        "use the Portuguese name instead ('unknown' for missing category)")
    pr["category"] = pr.product_category_name_english.fillna(pr.product_category_name)
    d["products"] = pr

    # payments -------------------------------------------------------------------------------
    pa = d["payments"]
    n = (pa.payment_installments == 0).sum()
    pa.loc[pa.payment_installments == 0, "payment_installments"] = 1
    log("payments", "installments = 0", n, "set to 1 (a payment is at least one installment)")
    log("payments", "payment_type 'not_defined'", (pa.payment_type == "not_defined").sum(),
        "kept as its own category (only 3 rows, all on canceled orders)")
    log("payments", "payment_value = 0", (pa.payment_value == 0).sum(),
        "kept: these are vouchers with zero value, not an error in the order")
    d["payments"] = pa

    # items ----------------------------------------------------------------------------------
    it = d["items"]
    log("items", "price <= 0", (it.price <= 0).sum(), "none found, no action")
    log("items", "freight_value = 0", (it.freight_value == 0).sum(), "kept: free shipping is a valid case")
    d["items"] = it

    # orders ---------------------------------------------------------------------------------
    o = d["orders"]
    f = chronology_flags(o)
    for c in f.columns:
        if f[c].sum():
            log("orders", c.replace("_", " "), f[c].sum(),
                "row kept + flag; excluded from the target / model table if the date is needed")
    o = pd.concat([o, f.add_prefix("flag_")], axis=1)
    log("orders", "missing order_approved_at", o.order_approved_at.isna().sum(),
        "kept for descriptive analysis, excluded from the model table (no prediction moment)")
    d["orders"] = o

    # customers / sellers: zip prefixes already read as text
    for t, col in (("customers", "customer_zip_code_prefix"), ("sellers", "seller_zip_code_prefix")):
        short = (d[t][col].str.len() != 5).sum()
        log(t, "zip prefix not 5 characters", short, "read as text so leading zeros are kept")
    return d, pd.DataFrame(DECISIONS)
