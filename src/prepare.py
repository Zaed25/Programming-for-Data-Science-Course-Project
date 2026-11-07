"""Step 1: raw tables -> audit tables + data/processed/orders_full.csv and orders_model.csv"""
import pandas as pd

from config import PROCESSED, TABLES
from load_data import load_raw, profile, table_overview
from clean import clean, key_report, integrity_report
from build_dataset import build_order_table, model_table


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    before = profile(raw, "before")
    key_report(raw).to_csv(TABLES / "keys_check.csv", index=False)
    integrity_report(raw).to_csv(TABLES / "integrity_check.csv", index=False)

    data, decisions = clean(raw)
    after = profile(data, "after")
    pd.concat([before, after]).to_csv(TABLES / "profile_columns.csv", index=False)
    table_overview(pd.concat([before, after])).to_csv(TABLES / "profile_tables.csv", index=False)
    decisions.to_csv(TABLES / "cleaning_decisions.csv", index=False)

    full, merges = build_order_table(data)
    merges.to_csv(TABLES / "merge_log.csv", index=False)
    assert full.order_id.is_unique and len(full) == len(raw["orders"]), "order table must have one row per order"
    full.to_csv(PROCESSED / "orders_full.csv", index=False)

    mt, excl = model_table(full)
    excl.to_csv(TABLES / "model_exclusions.csv", index=False)
    mt.to_csv(PROCESSED / "orders_model.csv", index=False)

    print(decisions.to_string(index=False))
    print(merges.to_string(index=False))
    print(excl.to_string(index=False))
    print(mt.split.value_counts().to_string(), "\nlate rate:", round(mt.is_late.mean(), 4))


if __name__ == "__main__":
    main()
