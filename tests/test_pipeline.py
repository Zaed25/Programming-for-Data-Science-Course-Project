"""Automatic checks: keys, no duplicated orders after joins, correct totals, target definition,
forbidden fields. Run with:  pytest -q   (from the project root, after src/prepare.py)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import PROCESSED, FORBIDDEN, ID_COLS          # noqa: E402
from load_data import load_raw                            # noqa: E402
from clean import clean, key_report                       # noqa: E402
from build_dataset import make_target, aggregate_payments, build_order_table   # noqa: E402
from features import FEATURES, NUMERIC, CATEGORICAL       # noqa: E402


# ----------------------------------------------------------------------------- target (small hand-made cases)
def _orders(status, delivered, estimated):
    return pd.DataFrame({
        "order_status": status,
        "order_delivered_customer_date": pd.to_datetime(delivered),
        "order_estimated_delivery_date": pd.to_datetime(estimated),
    })


def test_target_same_day_is_on_time():
    # delivered late in the evening of the estimated day -> still on time (date parts compared)
    t = make_target(_orders(["delivered"], ["2018-03-10 23:50:00"], ["2018-03-10 00:00:00"]))
    assert t.iloc[0] == 0


def test_target_next_day_is_late():
    t = make_target(_orders(["delivered"], ["2018-03-11 00:05:00"], ["2018-03-10 00:00:00"]))
    assert t.iloc[0] == 1


def test_target_before_estimate_is_on_time():
    t = make_target(_orders(["delivered"], ["2018-03-02 10:00:00"], ["2018-03-10 00:00:00"]))
    assert t.iloc[0] == 0


def test_target_missing_date_is_unknown_not_on_time():
    t = make_target(_orders(["delivered"], [None], ["2018-03-10"]))
    assert np.isnan(t.iloc[0])


def test_target_not_delivered_is_unknown():
    t = make_target(_orders(["canceled", "shipped"], ["2018-03-01", None], ["2018-03-10", "2018-03-10"]))
    assert t.isna().all()


# ----------------------------------------------------------------------------- aggregation
def test_payment_aggregation_keeps_totals():
    p = pd.DataFrame({"order_id": ["a", "a", "b"], "payment_sequential": [1, 2, 1],
                      "payment_type": ["credit_card", "voucher", "boleto"],
                      "payment_installments": [3, 1, 1], "payment_value": [80.0, 20.0, 50.0]})
    agg = aggregate_payments(p)
    assert len(agg) == 2
    assert agg.loc["a", "payment_total"] == 100.0
    assert agg.loc["a", "n_payment_records"] == 2 and agg.loc["a", "uses_voucher"] == 1
    assert agg.payment_total.sum() == p.payment_value.sum()


# ----------------------------------------------------------------------------- real data
@pytest.fixture(scope="module")
def built():
    raw = load_raw()
    data, _ = clean(raw)
    full, merges = build_order_table(data)
    return raw, full, merges


def test_primary_keys_unique(built):
    raw, _, _ = built
    rep = key_report(raw)
    assert (rep.duplicate_keys == 0).all() and (rep.null_keys == 0).all()


def test_one_row_per_order_after_joins(built):
    raw, full, merges = built
    assert full.order_id.is_unique
    assert len(full) == raw["orders"].order_id.nunique()
    assert (merges.rows_before == merges.rows_after).all()


def test_totals_preserved(built):
    raw, full, _ = built
    assert np.isclose(full.total_price.sum(), raw["items"].price.sum())
    assert np.isclose(full.total_freight.sum(), raw["items"].freight_value.sum())
    assert np.isclose(full.payment_total.sum(), raw["payments"].payment_value.sum())
    assert full.n_items.sum() == len(raw["items"])


def test_model_table_saved_one_row_per_order():
    path = PROCESSED / "orders_model.csv"
    if not path.exists():
        pytest.skip("run src/prepare.py first")
    m = pd.read_csv(path, usecols=["order_id", "is_late", "order_status", "split"])
    assert m.order_id.is_unique
    assert set(m.is_late.unique()) <= {0, 1}
    assert (m.order_status == "delivered").all()
    assert set(m.split) <= {"train", "val", "test"}


def test_no_forbidden_or_id_columns_in_features():
    assert not set(FEATURES) & set(FORBIDDEN)
    assert not set(FEATURES) & set(ID_COLS)
    leaky_words = ("delivered", "status", "delay_days", "actual", "carrier")
    assert not [f for f in FEATURES if any(w in f for w in leaky_words)]
    assert set(FEATURES) == set(NUMERIC) | set(CATEGORICAL)
