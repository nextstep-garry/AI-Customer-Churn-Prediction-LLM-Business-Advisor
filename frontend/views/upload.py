"""Upload a dataset -> the backend runs the whole pipeline -> open it in the dashboard."""
from pathlib import Path

import pandas as pd
import streamlit as st

from churn_ui import theme
from churn_ui.client import ApiError

model = theme.require_backend()
client = theme.get_client()

TEMPLATE = ("CustomerId,Surname,CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,"
            "IsActiveMember,EstimatedSalary,Exited\n"
            "1001,Example,619,France,Female,42,2,0,1,1,1,101348.88,1\n"
            "1002,Sample,608,Spain,Female,41,1,83807.86,1,0,1,112542.58,0\n")

st.title("Upload a dataset")
st.caption("Upload customer records as a CSV. The backend cleans and validates them, scores every customer "
           "and builds the portfolio dashboard.")

with st.expander("What should the file contain?"):
    fields = client.schema()["fields"]
    st.markdown("**Required columns** (header names are matched loosely: `credit_score`, `Credit Score` and "
                "`CreditScore` are all fine)")
    rows = []
    for f in fields:
        rule = ", ".join(f["choices"]) if f["type"] == "categorical" else (
            f'{"> " if f["exclusive_min"] else ""}{f["min"]:,.0f} to {f["max"]:,.0f}')
        rows.append({"Column": f["name"], "Valid values": rule})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("**Optional:** `CustomerId`, `Surname` (stored, but hidden on screen unless you tick Show surnames) and `Exited` "
                "(1 = churned, 0 = stayed) which unlocks the model-performance view. Other columns are ignored.")
    st.download_button("Download a CSV template", TEMPLATE, "churn_template.csv", "text/csv")

up = st.file_uploader("CSV file", type=["csv"])
c1, c2 = st.columns([1, 1.4])
name = c1.text_input("Dataset name", value=Path(up.name).stem if up else "", placeholder="e.g. Q3 customers")
policy = c2.radio("Rows with invalid values", ["reject", "repair"], horizontal=True,
                  format_func=lambda p: {"reject": "Set aside and report (recommended)",
                                         "repair": "Repair with median / mode (as in training)"}[p],
                  help="Reject: invalid rows are excluded from scoring and listed in the Data quality tab. "
                       "Repair: bad cells are replaced by the most common (or median) valid value in this file "
                       "and the row is flagged as repaired.")

if st.button("Run pipeline", type="primary", disabled=up is None):
    try:
        with st.status("Running the pipeline…", expanded=True) as status:
            out = client.upload(up.name, up.getvalue(), name or Path(up.name).stem, policy)
            for s in out["stages"]:
                st.write(f"✅ **{s['label']}** — {s['rows_in']:,} → {s['rows_out']:,} rows · {s['detail']}")
            status.update(label="Pipeline complete", state="complete", expanded=False)
        st.session_state["dataset_id"] = out["dataset"]["id"]
        st.session_state["just_uploaded"] = out["dataset"]
    except ApiError as exc:
        theme.show_error(exc, "The file could not be processed. ")

done = st.session_state.get("just_uploaded")
if done:
    r = done["rows"]
    m = st.columns(4)
    m[0].metric("Rows in file", f"{r['uploaded']:,}")
    m[1].metric("Customers scored", f"{r['scored']:,}")
    m[2].metric("Rejected / repaired", f"{r['rejected']:,} / {r['repaired']:,}")
    m[3].metric("Duplicates removed", f"{r['duplicates_removed']:,}")
    if st.button("Open the dashboard →", type="primary"):
        st.switch_page("views/portfolio.py")

st.divider()
st.subheader("Your datasets")
datasets = client.datasets()
if not datasets:
    st.write("No datasets yet.")
else:
    st.dataframe(pd.DataFrame([{
        "Name": d["name"], "Uploaded (UTC)": d["created_at"].replace("T", " ")[:16], "Scored": d["rows"]["scored"],
        "Rejected": d["rows"]["rejected"], "Repaired": d["rows"]["repaired"], "Labelled": d["has_target"],
        "Model": f"v{d['model']['version']}"} for d in datasets]), hide_index=True, width="stretch")
    pick = st.selectbox("Manage a dataset", [d["id"] for d in datasets],
                        format_func=lambda i: next(d["name"] for d in datasets if d["id"] == i))
    a, b, c = st.columns(3)
    if a.button("Open in dashboard", width="stretch"):
        st.session_state["dataset_id"] = pick
        st.switch_page("views/portfolio.py")
    if b.button("Re-score with the current model", width="stretch",
                help="Re-runs the pipeline on the stored file, e.g. after the model files were replaced."):
        try:
            client.rescore(pick)
            st.success("Re-scored.")
        except ApiError as exc:
            theme.show_error(exc)
    confirm = c.checkbox("Confirm delete")
    if c.button("Delete", width="stretch", disabled=not confirm):
        client.delete_dataset(pick)
        st.session_state.pop("dataset_id", None)
        st.rerun()
