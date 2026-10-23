import json
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from src import chatbot as cb
from src import forecasting as fc
from src.config import ForecastSettings


@pytest.fixture
def ctx(clean_df, series, settings):
    spec = fc.ModelSpec("ARIMAX", (1, 0, 0), trend="c", exog=("is_oct", "is_nov"), exog_name="oct_nov")
    sel = NS(series=series, selected=spec, settings=settings, family_best=pd.DataFrame(
        {"family": ["ARIMAX"], "model": [spec.label], "status": ["ok"], "wape": [0.1]}),
        test_table=pd.DataFrame({"model": [spec.label], "wape": [0.2]}), rationale="because", notes=["n1"])
    return cb.CopilotContext(clean_df, settings, sel)


def test_breakdown_matches_pandas(ctx):
    r = cb.run_tool(ctx, "breakdown", {"dimension": "country", "metric": "revenue", "top_n": 2})
    expected = ctx.base.groupby("country")["sales"].sum().sort_values(ascending=False)
    assert r["label"] == cb.HIST and r["rows"][0]["name"] == expected.index[0]
    assert r["rows"][0]["value"] == pytest.approx(expected.iloc[0], rel=1e-3)


def test_metrics_are_distinct(ctx):
    rev = cb.run_tool(ctx, "kpis", {})["revenue"]
    units = cb.run_tool(ctx, "breakdown", {"dimension": "year", "metric": "units"})
    orders = cb.run_tool(ctx, "breakdown", {"dimension": "year", "metric": "orders"})
    assert units["metric"] == "units" and orders["metric"] == "orders"
    assert sum(r["value"] for r in orders["rows"]) < sum(r["value"] for r in units["rows"]) < rev


def test_invalid_inputs_return_errors_not_exceptions(ctx):
    assert "error" in cb.run_tool(ctx, "breakdown", {"dimension": "DROP TABLE x"})
    assert "error" in cb.run_tool(ctx, "kpis", {"country": "Atlantis"})
    assert "error" in cb.run_tool(ctx, "nope", {})
    assert "error" in cb.run_tool(ctx, "kpis", {"bogus": 1})
    assert "error" in cb.run_tool(ctx, "compare_periods", {"period_a": "19", "period_b": "2020"})


def test_compare_periods_like_for_like(ctx):
    r = cb.run_tool(ctx, "compare_periods", {"period_a": "2018", "period_b": "2019"})
    assert r["label"] == cb.HIST and r["calendar_months_compared"] == list(range(1, 13))
    d = ctx.base
    a, b = d[d.year == 2018].sales.sum(), d[d.year == 2019].sales.sum()
    assert r["pct_change_b_vs_a"] == pytest.approx((b / a - 1) * 100, rel=1e-3)


def test_forecast_and_scenario_are_labelled_and_consistent(ctx):
    f = cb.run_tool(ctx, "forecast", {"horizon": 3})
    s = cb.run_tool(ctx, "scenario", {"adjustment_pct": -10, "horizon": 3})
    assert f["label"] == cb.FCST and s["label"] == cb.SCEN and len(f["rows"]) == 3
    assert s["rows"][0]["scenario"] == pytest.approx(f["rows"][0]["forecast"] * 0.9, rel=1e-3)


def test_forecast_without_models_gives_clear_message(clean_df, settings):
    r = cb.run_tool(cb.CopilotContext(clean_df, settings, None), "forecast", {})
    assert "Model Lab" in r["error"]


def test_results_are_json_serialisable(ctx):
    for name, args in [("dataset_overview", {}), ("monthly_trend", {"last_n": 6}), ("unusual_months", {}),
                       ("model_metrics", {}), ("kpis", {"year": 2019})]:
        json.dumps(cb.run_tool(ctx, name, args))


def test_router_maps_questions_to_tools():
    assert cb.route_question("forecast next 6 months")[0] == ("forecast", {"horizon": 6})
    assert cb.route_question("what if sales drop 15%")[0][1]["adjustment_pct"] == -15
    assert cb.route_question("compare 2018 and 2019")[0][0] == "compare_periods"
    assert cb.route_question("top countries by units")[0][1]["metric"] == "units"
    assert cb.route_question("which model is best")[0][0] == "model_metrics"


def test_demo_mode_without_key(ctx):
    cp = cb.Copilot(ctx, api_key="")
    assert cp.mode == "demo"
    a = cp.ask("top 3 countries by revenue")
    assert a.mode == "demo" and "HISTORICAL FACT" in a.text and a.tool_calls
    assert cp.ask("   ").error == "empty" and cp.ask("x" * 3000).error == "too_long"


class FakeGroq:
    """Minimal stand-in for groq.Groq: first call requests a tool, second answers."""

    def __init__(self, models=("llama-3.3-70b-versatile",), fail=False):
        self.calls, self.fail = [], fail
        self.models = NS(list=lambda: NS(data=[NS(id=m) for m in models]))
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        if self.fail:
            raise RuntimeError("boom")
        self.calls.append(kw)
        if len(self.calls) == 1:
            tc = NS(id="c1", function=NS(name="breakdown", arguments='{"dimension": "country", "top_n": 2}'))
            return NS(choices=[NS(message=NS(content="", tool_calls=[tc]))])
        tool_msgs = [m for m in kw["messages"] if m["role"] == "tool"]
        assert tool_msgs and "HISTORICAL FACT" in tool_msgs[0]["content"]     # LLM only sees tool output
        return NS(choices=[NS(message=NS(content="[HISTORICAL FACT] done", tool_calls=None))])


def test_groq_flow_uses_tools_and_system_prompt(ctx):
    fake = FakeGroq()
    cp = cb.Copilot(ctx, api_key="k", client=fake)
    a = cp.ask("top countries?", history=[{"role": "user", "content": "hi"}])
    assert a.mode == "groq" and a.text.startswith("[HISTORICAL FACT]") and a.tool_calls[0]["tool"] == "breakdown"
    first = fake.calls[0]
    assert first["messages"][0]["role"] == "system" and first["model"] == "llama-3.3-70b-versatile"
    assert len(json.dumps(first["messages"])) < 5000          # no dataset dump in the prompt


def test_groq_failure_falls_back_to_demo(ctx):
    a = cb.Copilot(ctx, api_key="k", client=FakeGroq(fail=True)).ask("top countries by revenue")
    assert a.mode == "demo" and a.error and "HISTORICAL FACT" in a.text


def test_model_resolution(ctx):
    assert cb.resolve_model(FakeGroq(models=("foo", "openai/gpt-oss-20b"))) == "openai/gpt-oss-20b"
    with pytest.raises(cb.CopilotError):
        cb.resolve_model(FakeGroq(models=("foo",)))
    with pytest.raises(cb.CopilotError):
        cb.resolve_model(FakeGroq(), env_model="retired-model")
    assert cb.resolve_model(FakeGroq(models=("a", "b")), env_model="b") == "b"


def test_missing_key_means_demo(monkeypatch, ctx):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(cb, "get_groq_key", lambda: None)
    assert cb.Copilot(ctx).mode == "demo"
