"""Dashboard 1: predictions for a whole uploaded dataset."""
import pandas as pd
import streamlit as st

from churn_ui import charts, theme
from churn_ui.client import ApiError

model = theme.require_backend()
client = theme.get_client()

SEGMENT_NAMES = {"Geography": "Country", "Gender": "Gender", "AgeBand": "Age band", "NumOfProducts": "Number of products",
                 "IsActiveMember": "Activity", "HasCrCard": "Credit card", "CreditBand": "Credit score band",
                 "BalanceBand": "Balance band", "TenureBand": "Tenure band"}


@st.cache_data(ttl=300, show_spinner="Loading dashboard…")
def load_summary(dataset_id: str, updated_at: str) -> dict:
    return client.summary(dataset_id)


@st.cache_data(ttl=60, show_spinner=False)
def load_customers(dataset_id: str, updated_at: str, **params) -> dict:
    return client.customers(dataset_id, **params)


@st.cache_data(ttl=300, show_spinner=False)
def load_csv(dataset_id: str, updated_at: str, kind: str) -> bytes:
    return client.download(dataset_id, kind)


st.title("Portfolio dashboard")
datasets = client.datasets()
if not datasets:
    st.info("No dataset yet. Upload a customer file and the whole portfolio is scored for you.")
    st.page_link("views/upload.py", label="Upload a dataset", icon="📤")
    st.stop()

ids = [d["id"] for d in datasets]
current = st.session_state.get("dataset_id")
label_of = {d["id"]: f'{d["name"]}  ·  {d["created_at"][:16].replace("T", " ")}  ·  {d["rows"]["scored"]:,} customers'
            for d in datasets}
sel = st.selectbox("Dataset", ids, index=ids.index(current) if current in ids else 0, format_func=label_of.get)
st.session_state["dataset_id"] = sel
meta = next(d for d in datasets if d["id"] == sel)
try:
    s = load_summary(sel, meta["updated_at"])
except ApiError as exc:
    theme.show_error(exc, "Could not load this dataset. ")
    st.stop()

k, m = s["kpis"], s["model"]
word, note = theme.score_words(m["score_type"])
if (meta["model"]["version"], meta["model"]["score_type"]) != (model["version"], model["score_type"]):
    st.warning(f"This dataset was scored with model v{meta['model']['version']} but the current model is "
               f"v{model['version']}. Use *Re-score* on the Upload page to refresh it.")

cols = st.columns(5)
cols[0].metric("Customers scored", f"{k['customers_scored']:,}", help=f"{k['rows_uploaded']:,} rows were in the file")
cols[1].metric("Flagged as likely to churn", f"{k['predicted_churners']:,}", f"{k['predicted_churn_rate']:.1%} of customers",
               delta_color="off")
cols[2].metric("High-risk customers", f"{k['high_risk']:,}")
cols[3].metric(f"Average {word.lower()}", theme.fmt_score(k["avg_score"], m["score_type"]))
cols[4].metric("Rejected / repaired rows", f"{k['rows_rejected']:,} / {k['rows_repaired']:,}",
               f"{k['duplicates_removed']:,} duplicates removed", delta_color="off")
st.caption(note)

t_over, t_seg, t_cust, t_drv, t_perf, t_qual = st.tabs(
    ["Overview", "Segments", "Customers", "Drivers", "Model performance", "Data quality"])

with t_over:
    a, b = st.columns(2)
    a.plotly_chart(charts.tier_bar(s["risk_tiers"]), width="stretch")
    b.plotly_chart(charts.score_hist(s["score_histogram"], m, word), width="stretch")
    cut = m["risk_tiers"]["cutoffs"]
    st.caption(f"Tiers: **Low** below {cut[0]:g}, **Medium** {cut[0]:g}–{cut[1]:g}, **High** {cut[1]:g} and above "
               f"(on the {m['score_type']} score). Customers at or above {m['threshold']:g} are flagged as likely to churn.")

with t_seg:
    by = st.selectbox("Break down by", list(s["segments"]), format_func=lambda x: SEGMENT_NAMES.get(x, x))
    rows = s["segments"][by]
    st.plotly_chart(charts.segment_bar(rows, f"Flag rate by {SEGMENT_NAMES.get(by, by).lower()}"), width="stretch")
    df = pd.DataFrame(rows).rename(columns={"segment": SEGMENT_NAMES.get(by, by), "customers": "Customers",
                                            "flagged": "Flagged", "avg_score": f"Avg {word.lower()}",
                                            "high_risk": "High risk", "flag_rate": "Flag rate",
                                            "actual_churn_rate": "Actual churn rate"})
    pct = {c: st.column_config.NumberColumn(format="percent") for c in ("Flag rate", "Actual churn rate") if c in df}
    st.dataframe(df, hide_index=True, width="stretch", column_config=pct)

with t_cust:
    f1, f2, f3, f4 = st.columns([1, 1, 1, 1.6])
    tier = f1.selectbox("Risk tier", ["All", *m["risk_tiers"]["labels"][::-1]])
    geo = f2.selectbox("Country", ["All", "France", "Germany", "Spain"])
    gender = f3.selectbox("Gender", ["All", "Female", "Male"])
    search = f4.text_input("Search customer id", "", help="Tick 'Show surnames' to also search by surname.")
    show_names = f4.checkbox("Show surnames", value=False,
                             help="Surnames are personal data: hidden by default, and only requested when this is ticked.")
    p1, p2, p3 = st.columns([1, 1, 2])
    size = p1.selectbox("Rows per page", [25, 50, 100], index=1)
    sort_by = p2.selectbox("Sort by", ["churn_score", "Age", "Balance", "CreditScore", "EstimatedSalary"],
                           format_func=lambda c: word if c == "churn_score" else c)
    query = dict(tier=tier, geography=geo, gender=gender, search=search, sort_by=sort_by, limit=size,
                 include_surname=show_names)
    try:
        first = load_customers(sel, meta["updated_at"], offset=0, **query)
    except ApiError as exc:
        if exc.code != "surname_not_allowed":
            raise
        st.warning("Surname access is disabled on this server, so surnames stay hidden.")
        show_names = False
        query["include_surname"] = False
        first = load_customers(sel, meta["updated_at"], offset=0, **query)
    pages = max(1, -(-first["total"] // size))
    page = p3.number_input(f"Page (of {pages:,})", 1, pages, 1)
    data = first if page == 1 else load_customers(sel, meta["updated_at"], offset=(page - 1) * size, **query)
    st.write(f"**{data['total']:,}** customers match")
    df = pd.DataFrame(data["rows"])
    if not df.empty:
        show = df.drop(columns=[c for c in ("raw_score", "data_issues", "outside_training_range") if c in df])
        ev = st.dataframe(show, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                          column_config={"churn_score": st.column_config.ProgressColumn(
                              word, min_value=0, max_value=1, format="percent" if m["score_type"] == "calibrated" else "%.2f"),
                              "will_churn": st.column_config.CheckboxColumn("Flagged")})
        st.caption("Select a row to see why that customer scored the way they did.")
        if ev.selection.rows:
            row = df.iloc[ev.selection.rows[0]]
            try:
                detail = client.customer(sel, int(row["row_id"]), include_surname=show_names)
                cust = detail["customer"]
                st.subheader(f"Customer {cust.get('CustomerId', cust['row_id'])}"
                             + (f" — {cust['Surname']}" if cust.get("Surname") else ""))
                x, y = st.columns([1, 1.4])
                x.markdown(f'{theme.badge(cust["risk_level"])} &nbsp; **{word}: '
                           f'{theme.fmt_score(cust["churn_score"], m["score_type"])}**', unsafe_allow_html=True)
                x.dataframe(pd.DataFrame({"Value": {k_: str(cust[k_]) for k_ in
                            ("CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts",
                             "HasCrCard", "IsActiveMember", "EstimatedSalary")}}), width="stretch")
                y.plotly_chart(charts.contribution_bar(detail["explanation"]["drivers"]), width="stretch")
            except ApiError as exc:
                theme.show_error(exc)
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("⬇ Predictions (CSV)", load_csv(sel, meta["updated_at"], "predictions"),
                       f"{meta['name']}_predictions.csv", "text/csv")
    if k["rows_rejected"]:
        d2.download_button("⬇ Rejected rows (CSV)", load_csv(sel, meta["updated_at"], "rejected"),
                           f"{meta['name']}_rejected.csv", "text/csv")

with t_drv:
    st.plotly_chart(charts.drivers_bar(s["drivers"]), width="stretch")
    st.caption("Average absolute SHAP value per feature over (up to 3,000 of) the scored customers. "
               "Bigger means that feature moves scores more.")

with t_perf:
    e = s["evaluation"]
    if not e:
        st.info("This file has no usable `Exited` column, so there is nothing to compare the predictions against. "
                "Add a column named `Exited` (1 = churned, 0 = stayed) to see accuracy, precision, recall and ROC.")
    else:
        st.info(e["note"])
        q = st.columns(5)
        for c_, (lbl, key) in zip(q, [("ROC-AUC", "roc_auc"), ("Precision", "precision"), ("Recall", "recall"),
                                      ("F1", "f1"), ("Accuracy", "accuracy")]):
            c_.metric(lbl, f"{e[key]:.3f}")
        a, b = st.columns(2)
        a.plotly_chart(charts.confusion(e["confusion"]), width="stretch")
        b.plotly_chart(charts.roc(e["roc_curve"], e["roc_auc"]), width="stretch")
        st.caption(f"{e['rows_evaluated']:,} labelled customers · actual churn rate {e['actual_churn_rate']:.1%}")

with t_qual:
    q = s["quality"]
    st.subheader("Pipeline steps")
    st.dataframe(pd.DataFrame(q["stages"])[["label", "rows_in", "rows_out", "detail"]].rename(
        columns={"label": "Step", "rows_in": "Rows in", "rows_out": "Rows out", "detail": "Result"}),
        hide_index=True, width="stretch")
    st.caption(f"Invalid-row policy: **{q['policy']}**. " + (f"Unused columns ignored: {', '.join(q['unused_columns'])}."
                                                          if q["unused_columns"] else ""))
    a, b = st.columns(2)
    with a:
        st.subheader("Invalid values by column")
        if q["issues_by_field"]:
            st.dataframe(pd.DataFrame(q["issues_by_field"].items(), columns=["Column", "Cells"]),
                         hide_index=True, width="stretch")
        else:
            st.write("None found.")
    with b:
        st.subheader("Drift vs training data")
        st.dataframe(pd.DataFrame(q["drift"])[["feature", "training_mean", "upload_mean", "shift_std", "status"]].rename(
            columns={"feature": "Feature", "training_mean": "Training mean", "upload_mean": "This file",
                     "shift_std": "Shift (std devs)", "status": "Status"}), hide_index=True, width="stretch",
            column_config={"Training mean": st.column_config.NumberColumn(format="%.1f"),
                           "This file": st.column_config.NumberColumn(format="%.1f"),
                           "Shift (std devs)": st.column_config.NumberColumn(format="%+.2f")})
        st.caption("How far this file's averages sit from the training data. Large shifts mean scores may be less reliable.")
    if q["out_of_training_range_by_field"]:
        st.warning("Values outside the range seen in training (scores may be less reliable): "
                   + ", ".join(f"{c}: {n:,}" for c, n in q["out_of_training_range_by_field"].items()))
    if q["rejected_examples"]:
        st.subheader(f"Rejected rows (first {len(q['rejected_examples'])})")
        st.dataframe(pd.DataFrame(q["rejected_examples"]), hide_index=True, width="stretch")
