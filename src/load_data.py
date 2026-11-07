"""Read the raw tables with correct types and produce a profile (rows, columns, types, missing)."""
import pandas as pd

from config import RAW, TABLES_FILES, STR_COLS, ORDER_DATES


def load_raw():
    """Return a dict name -> DataFrame. Raw files are never modified."""
    data = {}
    for name, fname in TABLES_FILES.items():
        df = pd.read_csv(RAW / fname, dtype=STR_COLS.get(name))
        data[name] = df
    for c in ORDER_DATES:
        data["orders"][c] = pd.to_datetime(data["orders"][c], errors="coerce")
    for c in ("shipping_limit_date",):
        data["items"][c] = pd.to_datetime(data["items"][c], errors="coerce")
    return data


def profile(data, stage):
    """One row per (table, column): dtype and missing values. Used before and after cleaning."""
    rows = []
    for name, df in data.items():
        for col in df.columns:
            rows.append({"stage": stage, "table": name, "rows": len(df), "columns": df.shape[1],
                         "column": col, "dtype": str(df[col].dtype),
                         "missing": int(df[col].isna().sum()),
                         "missing_pct": round(100 * df[col].isna().mean(), 2)})
    return pd.DataFrame(rows)


def table_overview(prof):
    """Short version: one line per table."""
    g = prof.groupby(["stage", "table"], sort=False)
    return g.agg(rows=("rows", "first"), columns=("columns", "first"),
                 cols_with_missing=("missing", lambda s: int((s > 0).sum())),
                 missing_cells=("missing", "sum")).reset_index()
