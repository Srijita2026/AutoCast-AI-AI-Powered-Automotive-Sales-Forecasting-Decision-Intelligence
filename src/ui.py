"""Streamlit helpers: theme, cached data/model loaders, global sidebar settings.

Caching design
--------------
* ``load_data``       keyed by (csv path, mtime)           -> re-runs only if the file changes.
* ``get_series``      keyed by (path, mtime, settings)     -> cheap.
* ``get_selection``   keyed by (path, mtime, settings)     -> the expensive model comparison (~25 s).
* ``get_forecast``    keyed by (..., spec, horizon)        -> cheap refit.
Page-level filters (Sales Analytics etc.) are not part of any cache key, so they never retrain models.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from . import data as dm
from . import diagnostics as dg
from . import forecasting as fc
from .config import RAW_DIR, TARGET_LABELS, ForecastSettings, setup_logging

setup_logging()
log = logging.getLogger(__name__)

CSS = """
<style>
.block-container {padding-top: 1.4rem; max-width: 1280px;}
.hero {background: linear-gradient(120deg,#1a1f2e 0%,#2a1410 60%,#ff4b2b33 100%); border:1px solid #2b3347;
       border-radius: 14px; padding: 18px 24px; margin-bottom: 14px;}
.hero h1 {margin:0; font-size: 1.7rem; letter-spacing:.3px;}
.hero p {margin:.2rem 0 0 0; color:#aab3c5;}
.kpi {background:#131a26; border:1px solid #232c40; border-left:4px solid #ff4b2b; border-radius:10px;
      padding:12px 14px; min-height:88px;}
.kpi .l {font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; color:#8b95a7;}
.kpi .v {font-size:1.55rem; font-weight:700; line-height:1.25;}
.kpi .s {font-size:.75rem; color:#8b95a7;}
.tag {display:inline-block; padding:1px 9px; border-radius:999px; font-size:.72rem; font-weight:600; margin-right:6px;}
.tag.hist {background:#16324a; color:#7cc4ff;} .tag.fc {background:#3b1d16; color:#ff9a84;}
.tag.sc {background:#3b3216; color:#ffd36b;}
@media (max-width: 800px) {.kpi .v {font-size:1.2rem;} .hero h1 {font-size:1.3rem;}}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    inject_css()
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def tag(kind: str) -> str:
    """Inline label chip: hist | fc | sc."""
    txt = {"hist": "HISTORICAL FACT", "fc": "MODEL FORECAST", "sc": "SCENARIO ASSUMPTION"}[kind]
    return f'<span class="tag {kind}">{txt}</span>'


def fmt_target(v: float, target: str) -> str:
    """Format a value according to the forecast target (money for revenue, counts otherwise)."""
    return money(v) if target == "revenue" else f"{v:,.0f}"


def target_name(target: str) -> str:
    return {"revenue": "Revenue ($)", "units": "Units", "orders": "Orders"}[target]


def money(v: float) -> str:
    return f"${v / 1e6:,.2f}M" if abs(v) >= 1e6 else f"${v:,.0f}"


# ----------------------------------------------------------------------------- data + settings
def csv_path() -> Path | None:
    return dm.find_csv(RAW_DIR)


def _mtime(p: Path) -> float:
    return p.stat().st_mtime


@st.cache_data(show_spinner="Loading and cleaning data...")
def _load(path: str, mtime: float):
    raw, clean, log_, aud = dm.load_and_clean(path, save=True)
    return raw, clean, log_, aud


def get_settings() -> ForecastSettings:
    """Applied settings (from the sidebar form); defaults when running a page standalone."""
    return st.session_state.get("settings", ForecastSettings())


def ensure_data() -> dict:
    """Load data or show a friendly placement/upload screen and stop the page."""
    p = csv_path()
    if p is None:
        inject_css()
        st.error("No dataset found.")
        st.markdown(f"Place **Auto_Sales_data.csv** (Kaggle: *ddosad/auto-sales-data*) in `{RAW_DIR}` "
                    "or upload it below, then reload.")
        up = st.file_uploader("Upload Auto_Sales_data.csv", type=["csv"])
        if up is not None:
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            (RAW_DIR / (Path(up.name).name or "Auto_Sales_data.csv")).write_bytes(up.getvalue())
            st.success("Saved. Reloading...")
            st.rerun()
        st.stop()
    try:
        raw, clean, log_, aud = _load(str(p), _mtime(p))
    except dm.DataLoadError as exc:
        st.error(f"Could not load the dataset: {exc}")
        st.stop()
    except Exception as exc:  # noqa: BLE001
        log.exception("load failed")
        st.error(f"Unexpected problem while loading the data: {exc}")
        st.stop()
    return {"path": str(p), "mtime": _mtime(p), "raw": raw, "df": clean, "log": log_, "audit": aud,
            "settings": get_settings()}


@st.cache_data(show_spinner=False)
def _series(path: str, mtime: float, settings: ForecastSettings):
    _, clean, _, _ = _load(path, mtime)
    return dg.build_series(clean, settings)


def get_series(d: dict) -> tuple[pd.Series, list[str]]:
    try:
        return _series(d["path"], d["mtime"], d["settings"])
    except ValueError as exc:
        st.error(f"Cannot build the time series: {exc}")
        st.stop()


@st.cache_data(show_spinner=False)
def _selection(path: str, mtime: float, settings: ForecastSettings):
    y, _ = _series(path, mtime, settings)
    return fc.run_selection(y, settings)


def get_selection(d: dict, run: bool = True):
    """Return the cached SelectionResult. If run=False, only return it when already computed this session."""
    key = (d["path"], d["mtime"], d["settings"])
    if not run and st.session_state.get("models_key") != key:
        return None
    try:
        with st.spinner("Comparing ARIMA / SARIMA / ARIMAX / SARIMAX with rolling-origin validation (about 30 s the first time)..."):
            sel = _selection(*key)
    except fc.InsufficientDataError as exc:
        st.error(f"Not enough data for the chosen validation design: {exc}")
        st.stop()
    except Exception as exc:  # noqa: BLE001
        log.exception("selection failed")
        st.error(f"Model comparison failed: {exc}")
        st.stop()
    st.session_state["models_key"] = key
    return sel


@st.cache_data(show_spinner=False)
def _forecast(path: str, mtime: float, settings: ForecastSettings, spec: fc.ModelSpec, horizon: int):
    y, _ = _series(path, mtime, settings)
    out, ff = fc.forecast_future(y, spec, horizon, settings.alpha)
    return out, ff.resid, ff.aic, ff.bic


def get_forecast(d: dict, spec: fc.ModelSpec, horizon: int):
    try:
        return _forecast(d["path"], d["mtime"], d["settings"], spec, horizon)
    except fc.FitFailure as exc:
        st.error(f"Forecast not available for {spec.label}: {exc}")
        return None


def base_df(d: dict) -> pd.DataFrame:
    """Transactions after the global status exclusions (same basis as the forecasting target)."""
    ex = d["settings"].exclude_statuses
    df = d["df"]
    return df[~df["status"].isin(ex)] if ex else df


# ----------------------------------------------------------------------------- sidebar
def sidebar_settings() -> None:
    """Global model/basis settings in a form so that changing widgets does not retrain until Apply is pressed."""
    p = csv_path()
    statuses = ["Cancelled", "Disputed", "In Process", "On Hold", "Resolved", "Shipped"]
    if p is not None:
        try:
            statuses = sorted(_load(str(p), _mtime(p))[1]["status"].unique())
        except Exception:  # noqa: BLE001
            pass
    cur = get_settings()
    with st.sidebar:
        st.markdown("### AutoCast AI")
        st.caption("Sales forecasting & analytics")
        with st.form("settings_form"):
            st.markdown("**Forecast & revenue basis**")
            target = st.selectbox("Target", list(TARGET_LABELS), index=list(TARGET_LABELS).index(cur.target),
                                  format_func=lambda k: TARGET_LABELS[k])
            excl = st.multiselect("Exclude statuses", statuses, default=[s for s in cur.exclude_statuses if s in statuses])
            test_p = st.slider("Hold-out months (untouched until selection)", 2, 6, cur.test_periods)
            metric = st.selectbox("Selection metric", ["wape", "rmse", "mae"], index=["wape", "rmse", "mae"].index(cur.primary_metric))
            min_train = st.slider("First training window (months)", 12, 22, cur.min_train_periods)
            go = st.form_submit_button("Apply settings", width="stretch")
        if go:
            st.session_state["settings"] = ForecastSettings(target=target, exclude_statuses=tuple(excl),
                                                            test_periods=test_p, primary_metric=metric,
                                                            min_train_periods=min_train)
            st.rerun()
        st.caption("Changing these retrains the models the next time they are needed. Page filters never do.")
