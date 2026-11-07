"""Model inputs (all known when the order is approved) and the preprocessing pipeline.
Imputation, scaling and encoding are fitted on the training split only (inside the Pipeline)."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# name: (source, how it is computed, why it may matter)
FEATURE_DICT = {
    "n_items": ("items", "number of item rows in the order", "more items, more handling"),
    "n_products": ("items", "distinct products", "mixed orders"),
    "n_sellers": ("items", "distinct sellers", "multi-seller orders ship in several packages"),
    "multi_seller": ("items", "1 if n_sellers > 1", "explicit flag for multi-seller orders"),
    "n_categories": ("items+products", "distinct categories", "order complexity"),
    "total_price": ("items", "sum of item prices", "order value"),
    "total_freight": ("items", "sum of freight values", "freight reflects distance and size"),
    "freight_ratio": ("items", "freight / (price + freight)", "expensive shipping relative to value = far/heavy"),
    "max_item_price": ("items", "most expensive item", "valuable items may be handled differently"),
    "total_weight_g": ("items+products", "sum of product weights", "heavy orders"),
    "max_weight_g": ("items+products", "heaviest product", "one bulky item can slow the order"),
    "total_volume_cm3": ("items+products", "sum of length*height*width", "bulky orders"),
    "planned_days": ("orders", "estimated date - approval date (days)", "the promise given to the customer"),
    "approval_delay_h": ("orders", "approval - purchase (hours)", "slow payment approval (e.g. boleto)"),
    "approve_weekday": ("orders", "day of week of approval", "weekend approvals wait longer"),
    "approve_hour": ("orders", "hour of approval", "late approvals ship next day"),
    "approve_month": ("orders", "month of approval", "seasonality (Black Friday, Christmas)"),
    "all_sellers_same_state": ("items+sellers+customers", "1 if every seller is in the customer state",
                               "short routes; handles multi-seller orders"),
    "main_seller_same_state": ("items+sellers+customers", "main seller state == customer state", "short route"),
    "n_seller_states": ("items+sellers", "distinct seller states", "packages come from several states"),
    "payment_total": ("payments", "sum of payment values", "order value incl. interest"),
    "n_payment_records": ("payments", "number of payment rows", "split payments"),
    "max_installments": ("payments", "max installments", "payment behaviour"),
    "uses_voucher": ("payments", "1 if any voucher", "payment behaviour"),
    "seller_prior_orders": ("history", "main seller's orders delivered before this approval", "seller experience"),
    "seller_prior_late_rate": ("history", "late share of those past orders (NaN if none)", "seller reliability"),
    "customer_prior_orders": ("history", "orders by the same person before this purchase", "repeat customer"),
    "customer_state": ("customers", "state of the customer", "remote states (north/north-east)"),
    "main_seller_state": ("items+sellers", "state of the main seller", "origin of the package"),
    "main_category": ("items+products", "category of the most expensive item", "product type"),
    "main_payment_type": ("payments", "payment type with the largest value", "boleto vs card"),
}
CATEGORICAL = ["customer_state", "main_seller_state", "main_category", "main_payment_type"]
NUMERIC = [f for f in FEATURE_DICT if f not in CATEGORICAL]
FEATURES = NUMERIC + CATEGORICAL


def preprocessor(scale=True):
    num = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num.append(("scale", StandardScaler()))
    cat = [("impute", SimpleImputer(strategy="constant", fill_value="missing")),
           ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                                    sparse_output=False))]
    return ColumnTransformer([("num", Pipeline(num), NUMERIC), ("cat", Pipeline(cat), CATEGORICAL)])
