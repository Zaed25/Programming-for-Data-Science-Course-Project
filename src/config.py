"""Paths and settings shared by every step of the pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

SEED = 42

# the 7 tables of the course pack (geolocation and reviews are not part of it)
TABLES_FILES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "translation": "product_category_name_translation.csv",
}

# zip code prefixes must stay text (leading zeros, e.g. "01046")
STR_COLS = {
    "customers": {"customer_zip_code_prefix": str},
    "sellers": {"seller_zip_code_prefix": str},
}

ORDER_DATES = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
               "order_delivered_customer_date", "order_estimated_delivery_date"]

# temporal split on order_approved_at (the prediction moment)
TRAIN_END = "2018-01-01"   # train: approved before this date
VAL_END = "2018-05-01"     # validation: [TRAIN_END, VAL_END), test: VAL_END and later
TEST_END = "2018-09-01"    # last month with enough delivered orders

TARGET = "is_late"

# information that only exists after approval -> never a model input
FORBIDDEN = ["order_status", "order_delivered_customer_date", "order_delivered_carrier_date",
             "actual_delivery_days", "delay_days", TARGET]
ID_COLS = ["order_id", "customer_id", "customer_unique_id", "main_seller_id"]
