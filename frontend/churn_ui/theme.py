"""Look and feel, plus the small helpers every page shares."""
from __future__ import annotations

import html

import streamlit as st

from .client import ApiClient, ApiError

RISK_COLORS = {"Low": "#2e9d6b", "Medium": "#e0a100", "High": "#d64545"}

_CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
.risk-card {border-radius: 14px; padding: 1.1rem 1.3rem; border: 1px solid rgba(128,128,128,.25);}
.risk-card h2 {margin: 0 0 .25rem 0; font-size: 1.5rem;}
.risk-card p {margin: 0; opacity: .85;}
.badge {display:inline-block; padding: .15rem .65rem; border-radius: 999px; color: #fff; font-weight: 600; font-size: .85rem;}
div[data-testid="stMetric"] {background: rgba(128,128,128,.07); padding: .7rem .9rem; border-radius: 12px;}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def badge(tier: str) -> str:
    """HTML pill for a risk tier. Values come from the backend, so they are escaped."""
    return f'<span class="badge" style="background:{RISK_COLORS.get(tier, "#888")}">{html.escape(str(tier))}</span>'


def score_words(score_type: str) -> tuple[str, str]:
    """(label, caveat). An uncalibrated model score must not be presented as a probability (audit 8.3b)."""
    if score_type == "calibrated":
        return "Churn probability", "Calibrated: of customers scored near X%, about X% actually churn."
    return ("Churn risk score",
            "A relative ranking score, not a probability: it runs about twice the real churn rate. "
            "Use it to compare and prioritise customers.")


def fmt_score(value: float, score_type: str) -> str:
    return f"{value:.0%}" if score_type == "calibrated" else f"{value:.2f}"


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient()


@st.cache_data(ttl=30, show_spinner=False)
def _ready() -> dict:
    return get_client().ready()


@st.cache_data(ttl=60, show_spinner=False)
def model_info() -> dict:
    return get_client().model()


def require_backend() -> dict:
    """Stop the page with a clear message unless the backend is up and its model is loaded."""
    try:
        state = _ready()
    except ApiError as exc:
        _ready.clear()
        st.error(f"**Backend unavailable.** {exc.message}")
        st.caption("Start it with `python wsgi.py` in the `backend/` folder, or check BACKEND_URL.")
        st.stop()
    if state.get("status") != "ready":
        _ready.clear()
        st.error("**The backend is running but its model did not load.**")
        st.code(state.get("detail") or "no detail provided", language="text")
        st.stop()
    return model_info()


def show_error(exc: ApiError, context: str = "") -> None:
    """Render an ApiError, with field-level details when present."""
    st.error(f"{context}{exc.message}")
    for d in exc.details:
        st.write(f"• **{d.get('field', '?')}** — {d.get('message', '')}")
    if exc.request_id:
        st.caption(f"Request id: `{exc.request_id}` (quote this when reporting a problem)")


def sidebar_status() -> None:
    """Small connection indicator for the sidebar."""
    from . import config
    try:
        state = _ready()
        if state.get("status") == "ready":
            mdl = state["model"]
            st.caption(f"🟢 Backend connected · {mdl['name']} v{mdl['version']}")
        else:
            st.caption("🟠 Backend up, model not loaded")
    except ApiError:
        _ready.clear()
        st.caption("🔴 Backend unreachable")
    st.caption(f"`{config.BACKEND_URL}`")
