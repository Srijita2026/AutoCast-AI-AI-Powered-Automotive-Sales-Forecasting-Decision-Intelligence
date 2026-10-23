"""Data discovery, auditing and cleaning. Raw files are never modified."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_CSV_NAME, PII_COLUMNS, PROCESSED_DIR, RAW_DIR

log = logging.getLogger(__name__)

ALIASES = {
    "ordernumber": "order_number", "quantityordered": "quantity_ordered", "priceeach": "price_each",
    "orderlinenumber": "order_line_number", "orderdate": "order_date",
    "dayssincelastorder": "days_since_last_order", "productline": "product_line",
    "productcode": "product_code", "customername": "customer_name", "addressline1": "address_line1",
    "postalcode": "postal_code", "contactlastname": "contact_last_name",
    "contactfirstname": "contact_first_name", "dealsize": "deal_size",
}
DATE_FORMATS = ("%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d")
CATEGORY_MAX_UNIQUE = 60


class DataLoadError(Exception):
    """Raised for user-fixable problems with the input data."""


# ----------------------------------------------------------------------------- discovery
def find_csv(raw_dir: Path = RAW_DIR) -> Path | None:
    """Return the preferred CSV in ``raw_dir`` (default file name first), else any CSV, else None."""
    if not raw_dir.exists():
        return None
    preferred = raw_dir / DEFAULT_CSV_NAME
    if preferred.exists():
        return preferred
    csvs = sorted(raw_dir.glob("*.csv"))
    return csvs[0] if csvs else None


def to_snake(name: str) -> str:
    """Normalise a column name to snake_case, applying readable aliases for known columns."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_").lower()
    return ALIASES.get(s.replace("_", ""), s)


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read a CSV as-is (no mutation). Tries utf-8 then latin-1."""
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"CSV not found: {path}. Place it in data/raw/ (see README).")
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
        except Exception as exc:  # malformed CSV
            raise DataLoadError(f"Could not parse {path.name}: {exc}") from exc
    else:
        raise DataLoadError(f"Could not decode {path.name} as utf-8 or latin-1.")
    if df.empty:
        raise DataLoadError(f"{path.name} contains no rows.")
    return df


def detect_date_format(values: pd.Series) -> str | None:
    """Pick the date format that parses the most values; day-first wins ties only if evidence exists."""
    sample = values.dropna().astype(str)
    best, best_ok = None, 0
    for fmt in DATE_FORMATS:
        ok = pd.to_datetime(sample, format=fmt, errors="coerce").notna().sum()
        if ok > best_ok:
            best, best_ok = fmt, ok
    if best is None or best_ok < 0.9 * len(sample):
        return None
    return best


def discover_schema(df: pd.DataFrame) -> dict:
    """Dynamically identify date, numeric, categorical columns and candidate sales targets."""
    date_cols, numeric_cols, cat_cols = [], [], []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            date_cols.append(c)
        elif pd.api.types.is_bool_dtype(s):
            continue
        elif pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(c)
        else:
            if "date" in c.lower() and detect_date_format(s) is not None:
                date_cols.append(c)
            elif s.nunique() <= CATEGORY_MAX_UNIQUE:
                cat_cols.append(c)
    pat = re.compile(r"sales|revenue|amount|quantity|qty|units", re.I)
    targets = [c for c in numeric_cols if pat.search(c) and not c.endswith("_diff")]
    return {"date_columns": date_cols, "numeric_columns": numeric_cols,
            "categorical_columns": cat_cols, "target_candidates": targets}


# ----------------------------------------------------------------------------- cleaning
def clean(raw: pd.DataFrame, drop_pii: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    """Clean a raw frame into an analysis-ready frame plus an audit log (list of change records).

    Policy: no blind zero-filling, no outlier deletion (outliers are *flagged*), exact duplicates dropped,
    rows lacking a valid date or sales value cannot be used for time series and are dropped (and logged).
    """
    log_: list[dict] = []

    def rec(step: str, detail: str, rows: int = 0) -> None:
        log_.append({"step": step, "detail": detail, "rows_affected": int(rows)})
        log.info("clean: %s | %s | %d", step, detail, rows)

    df = raw.copy()
    n0 = len(df)
    rename = {c: to_snake(c) for c in df.columns}
    df = df.rename(columns=rename)
    changed = {a: b for a, b in rename.items() if a != b}
    rec("rename_columns", f"{len(changed)} columns normalised to snake_case", 0)

    for c in df.select_dtypes(include=["object", "string"]).columns:
        before = df[c].copy()
        df[c] = df[c].astype("string").str.strip()
        n_changed = int((before.astype("string") != df[c]).fillna(False).sum())
        if n_changed:
            rec("strip_whitespace", f"{c}: trimmed whitespace", n_changed)

    schema = discover_schema(df)
    if not schema["date_columns"]:
        raise DataLoadError("No date column detected; a date field is required for forecasting.")
    date_col = "order_date" if "order_date" in schema["date_columns"] else schema["date_columns"][0]
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        fmt = detect_date_format(df[date_col])
        df[date_col] = pd.to_datetime(df[date_col], format=fmt, errors="coerce")
        rec("parse_dates", f"{date_col} parsed with explicit format {fmt}", len(df))
    if date_col != "order_date":
        df = df.rename(columns={date_col: "order_date"})

    for c in schema["numeric_columns"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "sales" not in df.columns:
        cands = schema["target_candidates"]
        raise DataLoadError(f"Expected a 'sales' column; candidates found: {cands}")

    bad = df["order_date"].isna() | df["sales"].isna()
    if bad.any():
        rec("drop_invalid_rows", "rows with unparseable date or missing sales removed", bad.sum())
        df = df.loc[~bad].copy()
    else:
        rec("missing_check", "no missing dates or sales values found; nothing imputed", 0)

    miss_other = df.drop(columns=["order_date", "sales"]).isna().sum()
    miss_other = miss_other[miss_other > 0]
    for c, k in miss_other.items():
        if df[c].dtype.kind in "OSU" or str(df[c].dtype) in ("string", "str"):
            df[c] = df[c].fillna("Unknown")
            rec("fill_missing_category", f"{c}: filled with 'Unknown'", k)
        else:
            rec("keep_missing_numeric", f"{c}: left as NaN (not zero-filled)", k)

    dups = df.duplicated()
    if dups.any():
        df = df.loc[~dups].copy()
        rec("drop_exact_duplicates", "fully identical rows removed (first kept)", dups.sum())
    else:
        rec("duplicates", "no exact duplicate rows", 0)
    if {"order_number", "order_line_number"} <= set(df.columns):
        kd = df.duplicated(["order_number", "order_line_number"]).sum()
        rec("key_duplicates", f"{kd} repeated (order_number, order_line_number) keys; kept for review", kd)

    # Derived columns and non-destructive flags
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["year_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    if {"quantity_ordered", "price_each"} <= set(df.columns):
        df["sales_check_diff"] = df["sales"] - df["quantity_ordered"] * df["price_each"]
        df["flag_sales_mismatch"] = df["sales_check_diff"].abs() > 0.5
        rec("flag_sales_mismatch", "sales != quantity*price (>0.50)", df["flag_sales_mismatch"].sum())
    if {"price_each", "msrp"} <= set(df.columns):
        df["price_to_msrp"] = df["price_each"] / df["msrp"].replace(0, np.nan)
        df["flag_price_high"] = df["price_to_msrp"] > 1.5
        df["flag_price_low"] = df["price_to_msrp"] < 0.5
        rec("flag_price_vs_msrp", "price >1.5x or <0.5x MSRP flagged, not altered",
            (df["flag_price_high"] | df["flag_price_low"]).sum())
    q1, q3 = df["sales"].quantile([0.25, 0.75])
    df["flag_sales_outlier"] = df["sales"] > q3 + 1.5 * (q3 - q1)
    rec("flag_sales_outliers", "line sales above Q3+1.5*IQR flagged; genuine large lines are kept",
        df["flag_sales_outlier"].sum())

    if drop_pii:
        pii = [c for c in PII_COLUMNS if c in df.columns]
        if pii:
            df = df.drop(columns=pii)
            rec("exclude_pii", f"contact columns excluded from processed data: {pii}", 0)
    df = df.sort_values(["order_date"] + (["order_number"] if "order_number" in df.columns else [])
                        ).reset_index(drop=True)
    rec("summary", f"{n0} raw rows -> {len(df)} clean rows", n0 - len(df))
    return df, log_


# ----------------------------------------------------------------------------- audit
def audit(clean_df: pd.DataFrame, raw: pd.DataFrame | None = None) -> dict:
    """Compute data-quality facts: shape, dtypes, ranges, missing, duplicates, categories, suspicious values."""
    out: dict = {"shape": clean_df.shape, "schema": discover_schema(clean_df)}
    out["dtypes"] = pd.DataFrame({"column": clean_df.columns, "dtype": clean_df.dtypes.astype(str).values,
                                  "n_unique": [clean_df[c].nunique() for c in clean_df.columns]})
    out["date_range"] = (clean_df["order_date"].min(), clean_df["order_date"].max())
    out["n_distinct_dates"] = int(clean_df["order_date"].nunique())
    src = raw if raw is not None else clean_df
    miss = src.isna().sum()
    out["missing"] = pd.DataFrame({"column": miss.index, "missing": miss.values,
                                   "pct": (miss.values / max(len(src), 1) * 100).round(2)})
    out["duplicates_exact"] = int(src.duplicated().sum())
    out["categories"] = {c: clean_df[c].value_counts().rename_axis(c).reset_index(name="rows")
                         for c in out["schema"]["categorical_columns"] if c in clean_df.columns}
    num = [c for c in out["schema"]["numeric_columns"] if c not in ("year", "month")]
    out["numeric_summary"] = clean_df[num].describe().T.round(2)
    checks = []
    for name, col, note in [
        ("sales != quantity x price", "flag_sales_mismatch", "line total inconsistent with its inputs"),
        ("price > 1.5x MSRP", "flag_price_high", "unusually high price vs list price"),
        ("price < 0.5x MSRP", "flag_price_low", "unusually deep discount vs list price"),
        ("line sales outlier (IQR)", "flag_sales_outlier", "large genuine lines are retained"),
    ]:
        if col in clean_df.columns:
            checks.append({"check": name, "rows": int(clean_df[col].sum()),
                           "pct": round(100 * clean_df[col].mean(), 2), "note": note})
    if "quantity_ordered" in clean_df.columns:
        checks.append({"check": "non-positive quantity", "rows": int((clean_df["quantity_ordered"] <= 0).sum()),
                       "pct": 0.0, "note": "returns/zero quantities would need separate treatment"})
    checks.append({"check": "non-positive sales", "rows": int((clean_df["sales"] <= 0).sum()), "pct": 0.0,
                   "note": "none expected for order lines"})
    out["suspicious"] = pd.DataFrame(checks)
    notes = []
    if "status" in clean_df.columns:
        non_ship = clean_df.loc[clean_df["status"] != "Shipped", "sales"].sum() / clean_df["sales"].sum()
        notes.append(f"{non_ship:.1%} of sales value belongs to non-'Shipped' statuses "
                     "(Cancelled/On Hold/Disputed/In Process/Resolved); target definition is configurable.")
    if "days_since_last_order" in clean_df.columns:
        notes.append("days_since_last_order has unclear semantics (it is not a constant offset from a fixed "
                     "date) and would not be known for future periods, so it is not used as a regressor.")
    out["notes"] = notes
    return out


def save_outputs(clean_df: pd.DataFrame, audit_log: list[dict], processed_dir: Path = PROCESSED_DIR) -> dict:
    """Write cleaned data + audit log to data/processed (raw is untouched)."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    p_csv, p_log = processed_dir / "auto_sales_clean.csv", processed_dir / "cleaning_audit_log.json"
    clean_df.to_csv(p_csv, index=False)
    p_log.write_text(json.dumps(audit_log, indent=2))
    return {"clean_csv": p_csv, "audit_log": p_log}


def load_and_clean(path: str | Path, save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], dict]:
    """Convenience pipeline: raw -> clean -> audit -> save. Returns (raw, clean, log, audit)."""
    raw = load_raw(path)
    clean_df, log_ = clean(raw)
    aud = audit(clean_df, raw)
    if save:
        try:
            save_outputs(clean_df, log_)
        except OSError as exc:
            log.warning("Could not save processed outputs: %s", exc)
    return raw, clean_df, log_, aud
