"""Churn Predictor: Streamlit front end. Run with:  streamlit run streamlit_app.py

All data comes from the Flask backend (BACKEND_URL); this app holds no model or business logic.
"""
import streamlit as st

from churn_ui import theme

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")
theme.inject_css()

nav = st.navigation({
    "Dashboards": [st.Page("views/portfolio.py", title="Portfolio dashboard", icon="📊", default=True),
                   st.Page("views/single.py", title="Single customer", icon="🧑")],
    "Data": [st.Page("views/upload.py", title="Upload dataset", icon="📤")],
    "Assistant": [st.Page("views/assistant.py", title="Assistant", icon="🤖")],
    "About": [st.Page("views/model_info.py", title="Model", icon="🧾")],
})
with st.sidebar:
    st.markdown("### 📉 Churn Predictor")
    theme.sidebar_status()
nav.run()
