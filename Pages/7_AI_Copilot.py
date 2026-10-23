"""AI Copilot: grounded chat over deterministic tools (Groq LLM or demo mode)."""
import streamlit as st

from src import ui
from src.chatbot import Copilot, CopilotContext

d = ui.ensure_data()
s = d["settings"]
ui.hero("AI Copilot", "Ask about history, categories, trends, model metrics, forecasts and what-if scenarios. "
        "Numbers come from Python tools, never from the LLM's memory.")

sel = ui.get_selection(d, run=False)
key = (d["path"], d["mtime"], s, id(sel))
if st.session_state.get("copilot_key") != key:
    st.session_state["copilot"] = Copilot(CopilotContext(d["df"], s, sel), model=st.session_state.get("groq_model"))
    st.session_state["copilot_key"] = key
cp: Copilot = st.session_state["copilot"]

if cp.mode == "demo":
    st.warning("**Demo mode** - no `GROQ_API_KEY` found. Charts, models and downloads all work; the chat answers with "
               "deterministic tool output instead of LLM prose. Copy `.env.example` to `.env`, add your key and restart "
               "to enable the LLM.")
else:
    st.success("Groq connected" + (f" - model `{cp.model}`" if cp.model else " (model is verified on the first question)"))
if sel is None:
    st.info("Forecast and model-metric questions need the model comparison. Open **Model Lab** once (about 30 s), then return.")

st.markdown(ui.tag("hist") + ui.tag("fc") + ui.tag("sc") + " Every answer labels what is history, forecast or assumption.",
            unsafe_allow_html=True)
hist = st.session_state.setdefault("chat", [])
cols = st.columns(4)
examples = ["Top 5 countries by revenue", "Which product line is weakest?", "Compare 2019 and 2020",
            "Forecast the next 6 months", "Which model is best and why?", "What if sales drop 10% next 3 months?",
            "Any unusual months?", "Show the monthly trend"]
clicked = None
for i, ex in enumerate(examples):
    if cols[i % 4].button(ex, key=f"ex{i}", width="stretch"):
        clicked = ex

for m in hist:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("tools"):
            with st.expander("Tools used"):
                st.json(m["tools"])

q = st.chat_input("Ask about the sales data...") or clicked
if q:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ans = cp.ask(q, [{"role": m["role"], "content": m["content"]} for m in hist])
        st.markdown(ans.text)
        if ans.tool_calls:
            with st.expander("Tools used"):
                st.json(ans.tool_calls)
        if ans.error and ans.mode != "demo":
            st.caption(f"Issue: {ans.error}")
    st.session_state["groq_model"] = cp.model
    hist += [{"role": "user", "content": q}, {"role": "assistant", "content": ans.text, "tools": ans.tool_calls}]
if hist and st.button("Clear conversation"):
    st.session_state["chat"] = []
    st.rerun()
