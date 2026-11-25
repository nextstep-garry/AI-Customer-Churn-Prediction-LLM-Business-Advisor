"""Dashboard 2: score one customer from typed-in details (the original app's form, now validated,
explained and tiered)."""
import pandas as pd
import streamlit as st

from churn_ui import charts, theme
from churn_ui.client import ApiError

model = theme.require_backend()
client = theme.get_client()

DEFAULTS = dict(CreditScore=619, Geography="France", Gender="Female", Age=42, Tenure=2, Balance=0.0,
                NumOfProducts=1, HasCrCard=1, IsActiveMember=1, EstimatedSalary=101348.88)


@st.cache_data(ttl=300, show_spinner=False)
def load_schema() -> list[dict]:
    return client.schema()["fields"]


st.title("Single customer prediction")
st.caption("Enter one customer's details to estimate their risk of leaving the bank.")

with st.form("customer"):
    record, cols = {}, st.columns(2)
    for i, f in enumerate(load_schema()):
        with cols[i % 2]:
            name, default = f["name"], DEFAULTS[f["name"]]
            if f["type"] == "categorical":
                record[name] = st.selectbox(f["label"], f["choices"], index=f["choices"].index(default))
            elif f["type"] == "integer" and (f["min"], f["max"]) == (0, 1):
                record[name] = int(st.toggle(f["label"], value=bool(default)))
            elif f["type"] == "integer":
                record[name] = st.slider(f["label"], int(f["min"]), int(f["max"]), int(default))
            else:
                lo = 0.01 if f["exclusive_min"] else float(f["min"])
                record[name] = st.number_input(f["label"], min_value=lo, max_value=float(f["max"]),
                                               value=float(default), step=1000.0, format="%.2f")
    submitted = st.form_submit_button("Predict churn risk", type="primary", width="stretch")

if submitted:
    try:
        with st.spinner("Scoring…"):
            st.session_state["single_result"] = client.predict(record, explain=True)
        st.session_state.pop("single_error", None)
    except ApiError as exc:
        st.session_state["single_error"] = exc
        st.session_state.pop("single_result", None)

if "single_error" in st.session_state:
    theme.show_error(st.session_state["single_error"], "The backend rejected this input. ")

res = st.session_state.get("single_result")
if res:
    label, note = theme.score_words(res["score_type"])
    as_pct = res["score_type"] == "calibrated"
    left, right = st.columns([1, 1.2])
    with left:
        st.plotly_chart(charts.gauge(res["churn_score"], model, label, as_pct), width="stretch")
    with right:
        st.markdown(
            f'<div class="risk-card"><h2>{theme.badge(res["risk_level"])} &nbsp;{res["prediction_label"]}</h2>'
            f'<p>{label}: <b>{theme.fmt_score(res["churn_score"], res["score_type"])}</b> &nbsp;·&nbsp; '
            f'flag threshold {theme.fmt_score(res["threshold_used"], res["score_type"])}</p></div>',
            unsafe_allow_html=True)
        st.caption(note)
        for w in res["warnings"]:
            st.warning(w)
    if res.get("explanation"):
        st.plotly_chart(charts.contribution_bar(res["explanation"]["drivers"]), width="stretch")
        st.caption("Contributions are in log-odds of the model's raw score (TreeSHAP). "
                   "Red factors push this customer's risk up, green ones pull it down.")
    st.caption(f"Model: {res['model_name']} v{res['model_version']} · request `{res['request_id']}`")

with st.expander("Recent predictions"):
    try:
        hist = client.history(10)
    except ApiError as exc:
        theme.show_error(exc)
        hist = []
    if hist:
        df = pd.DataFrame([{"Time (UTC)": h["ts"].replace("T", " ")[:19], "Age": h["input"]["Age"],
                            "Country": h["input"]["Geography"], "Products": h["input"]["NumOfProducts"],
                            "Score": h["score"], "Tier": h["risk_level"], "Model": h["model_version"]} for h in hist])
        st.dataframe(df, hide_index=True, width="stretch")
        if st.button("Clear history"):
            client.clear_history()
            st.rerun()
    else:
        st.write("Nothing yet.")
