"""AI Copilot: deterministic data tools + (optional) Groq LLM that only explains tool results.

Safety/grounding rules implemented here
* All numbers come from the Python tools below (``TOOLS``); the LLM never computes metrics.
* Only aggregated results are sent to the LLM - never raw rows or the dataset.
* No arbitrary code/SQL execution: the LLM can only call whitelisted tools with validated arguments.
* Every tool result carries a ``label``: HISTORICAL FACT, MODEL FORECAST or SCENARIO ASSUMPTION.
* No API key -> demo mode (rule-based router over the same tools); the rest of the app is unaffected.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import eda
from .config import ForecastSettings, get_groq_key
from .forecasting import FitFailure, SelectionResult, forecast_future

log = logging.getLogger(__name__)

HIST, FCST, SCEN = "HISTORICAL FACT", "MODEL FORECAST", "SCENARIO ASSUMPTION"
PREFERRED_MODELS = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]
DIMENSIONS = {"country": "country", "product_line": "product_line", "customer": "customer_name",
              "customer_name": "customer_name", "deal_size": "deal_size", "status": "status", "city": "city",
              "year": "year"}
MAX_TOOL_ROUNDS = 4


@dataclass
class CopilotContext:
    """Read-only data the tools operate on."""

    df: pd.DataFrame                              # cleaned transactions
    settings: ForecastSettings = field(default_factory=ForecastSettings)
    selection: SelectionResult | None = None

    @property
    def base(self) -> pd.DataFrame:
        """Rows used for revenue-type answers: configured status exclusions applied (default: Cancelled)."""
        ex = self.settings.exclude_statuses
        return self.df[~self.df["status"].isin(ex)] if ex else self.df

    @property
    def basis(self) -> str:
        ex = self.settings.exclude_statuses
        return f"excludes statuses {list(ex)}" if ex else "includes all statuses"


# ----------------------------------------------------------------------------- helpers
def _py(x: Any) -> Any:
    """Make results JSON-safe (numpy/pandas -> python)."""
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, pd.DataFrame):
        return _py(x.to_dict("records"))
    if isinstance(x, (pd.Timestamp,)):
        return x.strftime("%Y-%m")
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _err(msg: str) -> dict:
    return {"error": msg}


def _metric(metric: str | None) -> str:
    m = (metric or "revenue").lower()
    if m not in eda.METRICS:
        raise ValueError(f"metric must be one of {list(eda.METRICS)}")
    return m


def _filter(ctx: CopilotContext, year: int | None = None, country: str | None = None,
            product_line: str | None = None) -> pd.DataFrame:
    d = ctx.base
    if year is not None:
        d = d[d["year"] == int(year)]
    for col, val in (("country", country), ("product_line", product_line)):
        if val:
            options = {v.lower(): v for v in ctx.df[col].dropna().unique()}
            if str(val).lower() not in options:
                raise ValueError(f"Unknown {col} '{val}'. Valid: {sorted(options.values())}")
            d = d[d[col] == options[str(val).lower()]]
    return d


# ----------------------------------------------------------------------------- tools
def tool_dataset_overview(ctx: CopilotContext) -> dict:
    """Shape, date range and available categories of the dataset."""
    df = ctx.df
    return {"label": HIST, "n_rows": len(df), "date_min": df["order_date"].min(), "date_max": df["order_date"].max(),
            "countries": int(df["country"].nunique()), "product_lines": sorted(df["product_line"].unique()),
            "statuses": sorted(df["status"].unique()), "deal_sizes": sorted(df["deal_size"].unique()),
            "years": sorted(int(y) for y in df["year"].unique()), "basis": ctx.basis,
            "definitions": "revenue=sum of line SALES; units=sum QUANTITYORDERED; orders=distinct order numbers"}


def tool_kpis(ctx: CopilotContext, year: int | None = None, country: str | None = None,
              product_line: str | None = None) -> dict:
    """Headline KPIs for an optional year/country/product line."""
    d = _filter(ctx, year, country, product_line)
    return {"label": HIST, "basis": ctx.basis, "filters": {"year": year, "country": country, "product_line": product_line},
            **eda.kpis(d)}


def tool_breakdown(ctx: CopilotContext, dimension: str, metric: str = "revenue", top_n: int = 5,
                   year: int | None = None, order: str = "top") -> dict:
    """Top or bottom categories by a metric."""
    dim = DIMENSIONS.get(str(dimension).lower())
    if dim is None:
        return _err(f"dimension must be one of {sorted(DIMENSIONS)}")
    if order not in ("top", "bottom"):
        return _err("order must be 'top' or 'bottom'")
    m = _metric(metric)
    d = _filter(ctx, year)
    t = eda.agg(d, dim, m)
    total = t["value"].sum()
    t = t.head(min(int(top_n), 15)) if order == "top" else t.tail(min(int(top_n), 15)).sort_values("value")
    t["share_pct"] = (t["value"] / total * 100).round(1) if total else np.nan
    return {"label": HIST, "basis": ctx.basis, "metric": m, "dimension": dim, "order": order, "year": year,
            "rows": t.rename(columns={dim: "name"})}


def tool_monthly_trend(ctx: CopilotContext, metric: str = "revenue", last_n: int = 12) -> dict:
    """Monthly values for the most recent ``last_n`` months, with MoM/YoY where valid."""
    m = _metric(metric)
    mt = eda.monthly_table(ctx.base)
    g = eda.growth_table(mt, m).tail(min(int(last_n), 36)).reset_index().rename(columns={"index": "month"})
    return {"label": HIST, "basis": ctx.basis, "metric": m, "rows": g,
            "note": "MoM/YoY are null where the comparison month is zero or missing."}


def _parse_period(p: str) -> tuple[int, int | None]:
    s = str(p).strip()
    if re.fullmatch(r"\d{4}", s):
        return int(s), None
    mt = re.fullmatch(r"(\d{4})-(\d{1,2})", s)
    if mt and 1 <= int(mt.group(2)) <= 12:
        return int(mt.group(1)), int(mt.group(2))
    raise ValueError(f"period '{p}' must look like 2019 or 2019-11")


def tool_compare_periods(ctx: CopilotContext, period_a: str, period_b: str, metric: str = "revenue",
                         like_for_like: bool = True) -> dict:
    """Compare two years or two months; for years, restrict to calendar months present in both (like-for-like)."""
    m = _metric(metric)
    (ya, ma), (yb, mb) = _parse_period(period_a), _parse_period(period_b)
    d = ctx.base
    da = d[(d["year"] == ya) & ((d["month"] == ma) if ma else True)]
    db = d[(d["year"] == yb) & ((d["month"] == mb) if mb else True)]
    if da.empty or db.empty:
        return _err("One of the periods has no data.")
    months_used = None
    if like_for_like and ma is None and mb is None:
        common = set(da["month"]) & set(db["month"])
        da, db = da[da["month"].isin(common)], db[db["month"].isin(common)]
        months_used = sorted(int(x) for x in common)
    va, vb = eda.agg(da.assign(_k=1), "_k", m)["value"].iloc[0], eda.agg(db.assign(_k=1), "_k", m)["value"].iloc[0]
    return {"label": HIST, "basis": ctx.basis, "metric": m, "period_a": period_a, "value_a": va,
            "period_b": period_b, "value_b": vb, "pct_change_b_vs_a": (vb / va - 1) * 100 if va else None,
            "calendar_months_compared": months_used}


def tool_unusual_months(ctx: CopilotContext, metric: str = "revenue") -> dict:
    """Months far from typical (robust z-score > 3.5). Flagged, not removed."""
    m = _metric(metric)
    u = eda.unusual_months(eda.monthly_table(ctx.base), m)
    return {"label": HIST, "basis": ctx.basis, "metric": m, "rows": u,
            "note": "Descriptive only; the data cannot establish why these months are unusual."}


def tool_model_metrics(ctx: CopilotContext) -> dict:
    """Validation and hold-out metrics of the compared models."""
    s = ctx.selection
    if s is None:
        return _err("Model comparison has not been run yet. Open the Model Lab page first.")
    return {"label": FCST, "selected_model": s.selected.label if s.selected else None, "rationale": s.rationale,
            "validation_by_family": s.family_best.drop(columns=["status"], errors="ignore"),
            "holdout_test": s.test_table, "notes": s.notes,
            "metric_definitions": "WAPE=sum|err|/sum|actual|; MAPE only when actuals are all positive"}


def tool_forecast(ctx: CopilotContext, horizon: int = 3) -> dict:
    """Forecast with 95% prediction intervals from the selected model."""
    s = ctx.selection
    if s is None or s.selected is None:
        return _err("No validated model is available yet. Open the Model Lab page to run the comparison.")
    try:
        fc, _ = forecast_future(s.series, s.selected, int(horizon), s.settings.alpha)
    except (FitFailure, ValueError) as exc:
        return _err(f"Forecast could not be produced: {exc}")
    return {"label": FCST, "model": s.selected.label, "target": s.settings.target, "horizon": int(horizon),
            "rows": fc.reset_index(), "interval": "95% prediction interval",
            "caveat": "Short history (2 yearly cycles); intervals were narrower than nominal in validation."}


def tool_scenario(ctx: CopilotContext, adjustment_pct: float, horizon: int = 3) -> dict:
    """What-if: apply a user-specified % uplift/decline to the model forecast (assumption, not a prediction)."""
    base = tool_forecast(ctx, horizon)
    if "error" in base:
        return base
    k = 1 + float(adjustment_pct) / 100
    rows = [{**r, "scenario": r["forecast"] * k, "scenario_lower": r["lower_95"] * k,
             "scenario_upper": r["upper_95"] * k} for r in base["rows"].to_dict("records")]
    return {"label": SCEN, "assumption": f"all forecast values multiplied by {k:.3f} ({adjustment_pct:+.1f}%)",
            "base_model": base["model"], "rows": pd.DataFrame(rows),
            "caveat": "A flat multiplier is a user assumption, not something the model estimated."}


TOOLS: dict[str, Callable[..., dict]] = {
    "dataset_overview": tool_dataset_overview, "kpis": tool_kpis, "breakdown": tool_breakdown,
    "monthly_trend": tool_monthly_trend, "compare_periods": tool_compare_periods,
    "unusual_months": tool_unusual_months, "model_metrics": tool_model_metrics,
    "forecast": tool_forecast, "scenario": tool_scenario,
}


def _fn(name: str, desc: str, props: dict, req: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req or []}}}


_METRIC = {"type": "string", "enum": list(eda.METRICS), "description": "revenue | units | orders"}
TOOL_SCHEMAS = [
    _fn("dataset_overview", "Dataset size, date range, categories, metric definitions.", {}),
    _fn("kpis", "Headline KPIs, optionally filtered.", {"year": {"type": "integer"}, "country": {"type": "string"},
                                                       "product_line": {"type": "string"}}),
    _fn("breakdown", "Top/bottom categories by metric.", {"dimension": {"type": "string", "enum": sorted(DIMENSIONS)},
                                                         "metric": _METRIC, "top_n": {"type": "integer"},
                                                         "year": {"type": "integer"},
                                                         "order": {"type": "string", "enum": ["top", "bottom"]}},
        ["dimension"]),
    _fn("monthly_trend", "Recent monthly values with MoM/YoY.", {"metric": _METRIC, "last_n": {"type": "integer"}}),
    _fn("compare_periods", "Compare two years ('2019') or months ('2019-11').",
        {"period_a": {"type": "string"}, "period_b": {"type": "string"}, "metric": _METRIC,
         "like_for_like": {"type": "boolean"}}, ["period_a", "period_b"]),
    _fn("unusual_months", "Months that are statistically unusual.", {"metric": _METRIC}),
    _fn("model_metrics", "Forecast model validation/hold-out metrics and the selected model.", {}),
    _fn("forecast", "Future forecast with 95% intervals from the selected model.", {"horizon": {"type": "integer"}}),
    _fn("scenario", "What-if percent adjustment applied to the forecast.",
        {"adjustment_pct": {"type": "number"}, "horizon": {"type": "integer"}}, ["adjustment_pct"]),
]


def run_tool(ctx: CopilotContext, name: str, args: dict | None) -> dict:
    """Validated dispatcher: unknown tools/bad arguments return {'error': ...} instead of raising."""
    fn = TOOLS.get(name)
    if fn is None:
        return _err(f"Unknown tool '{name}'.")
    try:
        return _py(fn(ctx, **(args or {})))
    except TypeError as exc:
        return _err(f"Bad arguments for {name}: {exc}")
    except ValueError as exc:
        return _err(str(exc))
    except Exception as exc:  # noqa: BLE001
        log.exception("tool %s failed", name)
        return _err(f"{name} failed: {exc}")


# ----------------------------------------------------------------------------- demo mode
def _num(s: str) -> int | None:
    m = re.search(r"(\d+)\s*(?:month|period|mo\b)", s) or re.search(r"next\s+(\d+)", s)
    return int(m.group(1)) if m else None


def route_question(q: str) -> list[tuple[str, dict]]:
    """Rule-based question -> tool calls (used in demo mode and as a fallback)."""
    s = q.lower()
    metric = "units" if re.search(r"\bunits?\b|quantity|volume", s) else "orders" if "order" in s and "revenue" not in s else "revenue"
    years = re.findall(r"\b(20\d{2})\b", s)
    calls: list[tuple[str, dict]] = []
    pct = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", s)
    if re.search(r"scenario|what if|what-if|suppose|assum", s) and pct:
        sign = -1 if re.search(r"drop|decline|decrease|fall|lower|down|reduc", s) and not pct.group(1).startswith("-") else 1
        calls.append(("scenario", {"adjustment_pct": sign * float(pct.group(1)), "horizon": _num(s) or 3}))
    elif re.search(r"forecast|predict|projection|next \d+|future|expect", s):
        calls.append(("forecast", {"horizon": _num(s) or 3}))
    elif re.search(r"model|accuracy|mape|rmse|wape|arima|sarima|validation|best model", s):
        calls.append(("model_metrics", {}))
    elif re.search(r"unusual|spike|outlier|anomal|peak|surge", s):
        calls.append(("unusual_months", {"metric": metric}))
    elif len(years) >= 2 or re.search(r"yoy|year.over.year|compare|versus| vs ", s) and years:
        if len(years) >= 2:
            calls.append(("compare_periods", {"period_a": years[0], "period_b": years[1], "metric": metric}))
        else:
            calls.append(("monthly_trend", {"metric": metric, "last_n": 24}))
    elif re.search(r"trend|monthly|over time|growth|recent", s):
        calls.append(("monthly_trend", {"metric": metric, "last_n": 12}))
    elif re.search(r"top|best|worst|bottom|highest|lowest|largest|biggest|which|by (country|product|customer|deal|city|status)", s):
        dim = ("country" if re.search(r"countr|geograph|region|market", s) else
               "product_line" if re.search(r"product|line|categor|vehicle|car|motorcycle", s) else
               "customer_name" if "customer" in s or "client" in s else
               "deal_size" if "deal" in s else "status" if "status" in s else "city" if "city" in s else "country")
        calls.append(("breakdown", {"dimension": dim, "metric": metric, "order": "bottom" if re.search(r"worst|bottom|lowest|smallest", s) else "top",
                                    **({"year": int(years[0])} if years else {})}))
    elif re.search(r"kpi|total|overall|how much|summary|overview|revenue|sales", s):
        calls.append(("kpis", {**({"year": int(years[0])} if years else {})}))
    else:
        calls.append(("dataset_overview", {}))
    return calls


def format_result(name: str, r: dict) -> str:
    """Plain-markdown rendering of a tool result (demo mode)."""
    if "error" in r:
        return f"**Note:** {r['error']}"
    head = f"**[{r.get('label', '')}]**" + (f" _(basis: {r['basis']})_" if r.get("basis") else "")
    body: list[str] = []
    rows = r.get("rows")
    if isinstance(rows, list) and rows:
        df = pd.DataFrame(rows)
        for c in df.columns:
            if df[c].dtype.kind == "f":
                df[c] = df[c].map(lambda v: "" if v is None or pd.isna(v) else f"{v:,.2f}")
        body.append("```\n" + df.to_string(index=False) + "\n```")
    extras = {k: (", ".join(map(str, v)) if isinstance(v, list) else v) for k, v in r.items()
              if k not in ("rows", "label", "basis", "notes", "caveat", "note") and not isinstance(v, dict)
              and not (isinstance(v, list) and v and isinstance(v[0], dict))}
    if extras:
        body.append("\n".join(f"- {k.replace('_', ' ')}: {v:,.2f}" if isinstance(v, float) else f"- {k.replace('_', ' ')}: {v}"
                              for k, v in extras.items()))
    for k in ("validation_by_family", "holdout_test"):
        if r.get(k):
            body.append(f"_{k.replace('_', ' ')}_\n```\n" + pd.DataFrame(r[k]).round(3).to_string(index=False) + "\n```")
    for k in ("caveat", "note", "notes"):
        if r.get(k):
            v = r[k] if isinstance(r[k], list) else [r[k]]
            body.append("\n".join(f"> {x}" for x in v))
    return head + "\n\n" + "\n\n".join(body)


# ----------------------------------------------------------------------------- Groq
SYSTEM_PROMPT = """You are AutoCast AI Copilot, an analyst assistant for an automotive/vehicle-models sales dataset.
Rules you must follow:
1. Every number you state MUST come from a tool result in this conversation. Never compute metrics, totals or
   forecasts yourself and never use numbers from memory. If a tool has not been called for a figure, call it.
2. Label statements by type: [HISTORICAL FACT] for past data, [MODEL FORECAST] for model output, and
   [SCENARIO ASSUMPTION] for what-if results. Keep these separate; never present a forecast or scenario as fact.
3. Describe patterns without claiming causes unless the data shows them; say when the data cannot explain something.
4. Mention caveats returned by tools (short history, interval width, basis/status exclusions).
5. If a question cannot be answered with the available tools, say so plainly. Be concise (under ~200 words)."""


class CopilotError(Exception):
    """User-presentable copilot failure (missing key, model unavailable, API error)."""


@dataclass
class ChatAnswer:
    text: str
    mode: str                                   # 'groq' | 'demo'
    tool_calls: list[dict] = field(default_factory=list)
    error: str | None = None


def resolve_model(client: Any, env_model: str | None = None) -> str:
    """Pick a currently supported chat model: GROQ_MODEL if available, else first preferred model that exists."""
    try:
        available = {m.id for m in client.models.list().data}
    except Exception as exc:  # noqa: BLE001
        raise CopilotError(f"Could not list Groq models ({type(exc).__name__}). Check your key and network.") from exc
    if env_model:
        if env_model in available:
            return env_model
        raise CopilotError(f"GROQ_MODEL '{env_model}' is not available to your key. Examples that are: "
                           f"{sorted(m for m in available if 'whisper' not in m)[:6]}")
    for m in PREFERRED_MODELS:
        if m in available:
            return m
    raise CopilotError("None of the preferred models are available: " + ", ".join(PREFERRED_MODELS))


class Copilot:
    """Chat orchestrator. Uses Groq when a key is configured, otherwise demo mode."""

    def __init__(self, ctx: CopilotContext, api_key: str | None = None, model: str | None = None,
                 client: Any = None) -> None:
        import os
        self.ctx = ctx
        self.api_key = api_key if api_key is not None else get_groq_key()
        self.env_model = model or os.environ.get("GROQ_MODEL") or None
        self._client = client
        self.model: str | None = None
        self.last_error: str | None = None

    @property
    def mode(self) -> str:
        return "groq" if (self.api_key or self._client) else "demo"

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from groq import Groq
            except ImportError as exc:
                raise CopilotError("The 'groq' package is not installed (pip install groq).") from exc
            self._client = Groq(api_key=self.api_key)
        return self._client

    def demo_answer(self, question: str, note: str | None = None) -> ChatAnswer:
        calls = route_question(question)
        parts, logged = [], []
        for name, args in calls:
            res = run_tool(self.ctx, name, args)
            logged.append({"tool": name, "args": args})
            parts.append(format_result(name, res))
        prefix = ("_Demo mode - answers are produced by deterministic tools without an LLM. "
                  "Add GROQ_API_KEY for natural-language explanations._\n\n") if note is None else f"_{note}_\n\n"
        return ChatAnswer(prefix + "\n\n".join(parts), "demo", logged)

    def ask(self, question: str, history: list[dict] | None = None) -> ChatAnswer:
        """Answer a question. Never raises; failures are returned in ``ChatAnswer.error``."""
        question = (question or "").strip()
        if not question:
            return ChatAnswer("Please type a question.", self.mode, error="empty")
        if len(question) > 2000:
            return ChatAnswer("Question too long (max 2000 characters).", self.mode, error="too_long")
        if self.mode == "demo":
            return self.demo_answer(question)
        try:
            return self._ask_groq(question, history or [])
        except CopilotError as exc:
            self.last_error = str(exc)
            fb = self.demo_answer(question, note=f"Groq unavailable: {exc} Showing deterministic result instead.")
            fb.error = str(exc)
            return fb
        except Exception as exc:  # noqa: BLE001
            log.exception("Groq call failed")
            msg = f"{type(exc).__name__}: {str(exc)[:200]}"
            self.last_error = msg
            fb = self.demo_answer(question, note=f"Groq request failed ({msg}). Showing deterministic result instead.")
            fb.error = msg
            return fb

    def _ask_groq(self, question: str, history: list[dict]) -> ChatAnswer:
        client = self._get_client()
        if self.model is None:
            self.model = resolve_model(client, self.env_model)
        msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        msgs += [{"role": m["role"], "content": m["content"]} for m in history[-8:] if m["role"] in ("user", "assistant")]
        msgs.append({"role": "user", "content": question})
        logged: list[dict] = []
        for _ in range(MAX_TOOL_ROUNDS):
            resp = client.chat.completions.create(model=self.model, messages=msgs, tools=TOOL_SCHEMAS,
                                                  tool_choice="auto", temperature=0.1, max_tokens=700)
            msg = resp.choices[0].message
            calls = getattr(msg, "tool_calls", None)
            if not calls:
                text = (msg.content or "").strip()
                if not logged:
                    text += "\n\n_Note: no data tool was called for this answer; verify any figures._"
                return ChatAnswer(text or "I could not produce an answer.", "groq", logged)
            msgs.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                        for c in calls]})
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                res = run_tool(self.ctx, c.function.name, args)
                logged.append({"tool": c.function.name, "args": args})
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(res)[:6000]})
        return ChatAnswer("I needed too many tool calls to answer. Try a more specific question.", "groq", logged,
                          error="max_tool_rounds")
