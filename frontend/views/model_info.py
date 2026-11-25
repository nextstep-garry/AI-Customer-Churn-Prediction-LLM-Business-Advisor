"""What model is running, how good is it, and what should you not trust it for."""
import pandas as pd
import streamlit as st

from churn_ui import theme

m = theme.require_backend()
word, note = theme.score_words(m["score_type"])

st.title("About the model")
st.markdown(f"**{m['name']}** · version **{m['version']}** · {m['score_type']} score · built {m['created_at']}")
st.caption(m["description"])

a, b, c = st.columns(3)
a.metric("Flag threshold", f"{m['threshold']:g}")
cut = m["risk_tiers"]["cutoffs"]
b.metric("Risk tier cut-offs", " / ".join(f"{x:g}" for x in cut))
c.metric("Score type", m["score_type"].title())
st.info(note)
st.caption(m["risk_tiers"]["note"])

st.subheader("Measured performance")
t = m["metrics"].get("test", {})
if t:
    cols = st.columns(5)
    for col, (lbl, key) in zip(cols, [("ROC-AUC", "roc_auc"), ("Precision", "precision"), ("Recall", "recall"),
                                      ("F1", "f1"), ("Accuracy", "accuracy")]):
        col.metric(lbl, f"{t[key]:.3f}")
    split = m["metrics"].get("split", {})
    st.caption(f"Held-out test set ({split.get('test', '?')} customers). {m['metrics'].get('source', '')}")

st.subheader("Known limitations")
for item in m["known_limitations"]:
    st.markdown(f"- {item}")

st.subheader("Environment")
rows = [{"Library": lib, "Built with": m["trained_with"].get(lib, "?"), "Installed": v,
         "Match": "✅" if m["trained_with"].get(lib, "").split(".")[:2] == v.split(".")[:2] else "⚠️"}
        for lib, v in m["installed"].items()]
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
with st.expander("Inputs, ranges seen in training, and artifact fingerprints"):
    st.write("Model features:", ", ".join(m["feature_names"]))
    st.json({"training_ranges": m["training_ranges"], "artifact_sha256_prefix": m["artifact_hashes"]})
