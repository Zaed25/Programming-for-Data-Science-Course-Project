"""Order-level table: aggregate items and payments separately, merge, create target + features.

Unit of analysis = one order. Items (many per order) and payments (many per order) are first
aggregated to one row per order, and only then merged, so prices are never duplicated.
"""
import numpy as np
import pandas as pd

from config import TRAIN_END, VAL_END, TEST_END

MERGE_LOG = []   # rows before/after each join


def _track(step, before, after, note=""):
    MERGE_LOG.append({"step": step, "rows_before": int(before), "rows_after": int(after), "note": note})


# ----------------------------------------------------------------------------- target
def make_target(orders):
    """is_late = 1 if the customer got the order on a later DAY than the estimated day,
    0 if on that day or before. NaN when the order is not delivered or a date is missing/invalid
    (we never turn unknown outcomes into 'on time')."""
    delivered = orders.order_delivered_customer_date
    est = orders.order_estimated_delivery_date
    ok = (orders.order_status == "delivered") & delivered.notna() & est.notna()
    if "flag_delivered_before_purchase" in orders:
        ok &= ~orders.flag_delivered_before_purchase
    late = (delivered.dt.normalize() > est.dt.normalize()).astype(float)   # compare date parts only
    return late.where(ok)


# ----------------------------------------------------------------------------- aggregation
def aggregate_items(items, products, sellers):
    n0 = len(items)
    it = items.merge(products[["product_id", "category", "product_weight_g", "product_volume_cm3",
                               "product_photos_qty"]], on="product_id", how="left", validate="many_to_one")
    it = it.merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left", validate="many_to_one")
    _track("items + products + sellers", n0, len(it), "many-to-one joins, row count must not change")

    g = it.groupby("order_id")
    agg = g.agg(n_items=("order_item_id", "count"), n_products=("product_id", "nunique"),
                n_sellers=("seller_id", "nunique"), n_categories=("category", "nunique"),
                total_price=("price", "sum"), total_freight=("freight_value", "sum"),
                max_item_price=("price", "max"), total_weight_g=("product_weight_g", "sum"),
                max_weight_g=("product_weight_g", "max"), total_volume_cm3=("product_volume_cm3", "sum"),
                n_seller_states=("seller_state", "nunique"))
    # sum() of all-NaN gives 0 -> put NaN back when no weight is known at all
    agg.loc[g.product_weight_g.count() == 0, "total_weight_g"] = np.nan
    agg.loc[g.product_volume_cm3.count() == 0, "total_volume_cm3"] = np.nan

    # main item / seller = the one with the largest value in the order (multi-seller orders)
    it["line_value"] = it.price + it.freight_value
    seller_val = it.groupby(["order_id", "seller_id", "seller_state"], as_index=False).line_value.sum()
    main = seller_val.sort_values(["order_id", "line_value", "seller_id"], ascending=[True, False, True]) \
                     .drop_duplicates("order_id")
    agg = agg.join(main.set_index("order_id")[["seller_id", "seller_state"]]
                   .rename(columns={"seller_id": "main_seller_id", "seller_state": "main_seller_state"}))
    cat = it.sort_values(["order_id", "price", "product_id"], ascending=[True, False, True]) \
            .drop_duplicates("order_id").set_index("order_id").category
    agg["main_category"] = cat
    agg["_seller_states"] = it.groupby("order_id").seller_state.agg(lambda s: set(s.dropna()))
    return agg


def aggregate_payments(payments):
    g = payments.groupby("order_id")
    agg = g.agg(payment_total=("payment_value", "sum"), n_payment_records=("payment_sequential", "count"),
                n_payment_types=("payment_type", "nunique"), max_installments=("payment_installments", "max"))
    agg["uses_voucher"] = g.payment_type.agg(lambda s: int((s == "voucher").any()))
    main = payments.sort_values(["order_id", "payment_value"], ascending=[True, False]).drop_duplicates("order_id")
    agg["main_payment_type"] = main.set_index("order_id").payment_type
    return agg


# ----------------------------------------------------------------------------- history features
def seller_history(df):
    """For the main seller: number of orders already DELIVERED before this order was approved and
    the share of them that were late. Only past information is used (merge_asof, strictly before)."""
    hist = df.loc[df.is_late.notna(), ["main_seller_id", "order_delivered_customer_date", "is_late"]] \
             .dropna(subset=["main_seller_id"]).sort_values("order_delivered_customer_date")
    hist["prior_n"] = hist.groupby("main_seller_id").cumcount() + 1
    hist["prior_late"] = hist.groupby("main_seller_id").is_late.cumsum()
    q = df[["order_id", "main_seller_id", "order_approved_at"]].dropna().sort_values("order_approved_at")
    m = pd.merge_asof(q, hist[["main_seller_id", "order_delivered_customer_date", "prior_n", "prior_late"]],
                      left_on="order_approved_at", right_on="order_delivered_customer_date",
                      by="main_seller_id", allow_exact_matches=False, direction="backward")
    m["prior_n"] = m.prior_n.fillna(0)
    m["seller_prior_late_rate"] = np.where(m.prior_n > 0, m.prior_late / m.prior_n.replace(0, np.nan), np.nan)
    return m.set_index("order_id")[["prior_n", "seller_prior_late_rate"]] \
            .rename(columns={"prior_n": "seller_prior_orders"})


def customer_history(df):
    """How many orders the same person (customer_unique_id) placed before this purchase."""
    s = df[["order_id", "customer_unique_id", "order_purchase_timestamp"]] \
        .sort_values(["customer_unique_id", "order_purchase_timestamp", "order_id"])
    s["customer_prior_orders"] = s.groupby("customer_unique_id").cumcount()
    return s.set_index("order_id").customer_prior_orders


# ----------------------------------------------------------------------------- main build
def build_order_table(d):
    MERGE_LOG.clear()
    o = d["orders"].copy()
    o["is_late"] = make_target(o)

    n = len(o)
    df = o.merge(d["customers"], on="customer_id", how="left", validate="one_to_one")
    _track("orders + customers", n, len(df), "one-to-one on customer_id")

    items_agg = aggregate_items(d["items"], d["products"], d["sellers"])
    n = len(df)
    df = df.merge(items_agg, left_on="order_id", right_index=True, how="left", validate="one_to_one")
    _track("orders + items (aggregated)", n, len(df), "items grouped to order level first")

    pay_agg = aggregate_payments(d["payments"])
    n = len(df)
    df = df.merge(pay_agg, left_on="order_id", right_index=True, how="left", validate="one_to_one")
    _track("orders + payments (aggregated)", n, len(df), "payments grouped to order level first")

    # ---- features known at approval time ------------------------------------------------
    ap, pu, est = df.order_approved_at, df.order_purchase_timestamp, df.order_estimated_delivery_date
    df["planned_days"] = (est.dt.normalize() - ap.dt.normalize()).dt.days
    df["approval_delay_h"] = (ap - pu).dt.total_seconds() / 3600
    df["approve_weekday"] = ap.dt.dayofweek
    df["approve_hour"] = ap.dt.hour
    df["approve_month"] = ap.dt.month
    df["multi_seller"] = (df.n_sellers > 1).astype(float).where(df.n_sellers.notna())
    df["freight_ratio"] = df.total_freight / (df.total_price + df.total_freight)
    # same state: share of the order's seller states equal to the customer state (handles multi-seller)
    df["all_sellers_same_state"] = [
        np.nan if not isinstance(ss, set) or not ss else float(ss == {cs})
        for ss, cs in zip(df._seller_states, df.customer_state)]
    df["main_seller_same_state"] = (df.main_seller_state == df.customer_state).astype(float) \
        .where(df.main_seller_state.notna())
    df = df.drop(columns="_seller_states")

    # ---- descriptive only (outcome information, NOT model inputs) ------------------------
    df["actual_delivery_days"] = (df.order_delivered_customer_date - pu).dt.total_seconds() / 86400
    df["delay_days"] = (df.order_delivered_customer_date.dt.normalize() - est.dt.normalize()).dt.days

    df = df.join(seller_history(df), on="order_id")
    df["customer_prior_orders"] = df.order_id.map(customer_history(df))

    # ---- split by approval time ---------------------------------------------------------
    df["split"] = np.select([ap < TRAIN_END, ap < VAL_END, ap < TEST_END], ["train", "val", "test"], "out_of_range")
    df.loc[ap.isna(), "split"] = "no_approval"
    return df, pd.DataFrame(MERGE_LOG)


def model_table(df):
    """Keep the orders that can be used for the 'predict at approval' task, logging every exclusion."""
    steps = [
        ("all orders", pd.Series(True, index=df.index)),
        ("status = delivered", df.order_status == "delivered"),
        ("delivered date present", df.order_delivered_customer_date.notna()),
        ("delivered not before purchase", ~df.flag_delivered_before_purchase),
        ("approval date present", df.order_approved_at.notna()),
        ("approval not before purchase", ~df.flag_approved_before_purchase),
        ("has at least one item", df.n_items.notna()),
        ("has payment records", df.n_payment_records.notna()),
        ("approved before " + TEST_END, df.order_approved_at < TEST_END),
    ]
    keep = pd.Series(True, index=df.index)
    log = []
    for name, cond in steps:
        before = int(keep.sum())
        keep &= cond.fillna(False)
        log.append({"filter": name, "rows_left": int(keep.sum()), "removed": before - int(keep.sum())})
    log[0]["removed"] = 0
    return df[keep].copy(), pd.DataFrame(log)
