"""Plotly figures. Pure functions of backend JSON, no Streamlit calls."""
from __future__ import annotations

import plotly.graph_objects as go

from .theme import RISK_COLORS

_LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
               font=dict(size=13))
_TIER_TINT = {"Low": "#d7f0e3", "Medium": "#faecc4", "High": "#f6d3d3"}


def _tier_edges(labels: list[str], cutoffs: list[float]):
    edges = [0.0, *cutoffs, 1.0]
    return list(zip(labels, edges[:-1], edges[1:]))


def gauge(score: float, model: dict, title: str, as_percent: bool) -> go.Figure:
    tiers = model["risk_tiers"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, title={"text": title},
        number={"valueformat": ".0%" if as_percent else ".2f"},
        gauge={"axis": {"range": [0, 1], "tickformat": ".0%" if as_percent else ".1f"},
               "bar": {"color": "#33475b", "thickness": 0.25},
               "steps": [{"range": [a, b], "color": _TIER_TINT.get(t, "#eee")}
                         for t, a, b in _tier_edges(tiers["labels"], tiers["cutoffs"])],
               "threshold": {"line": {"color": "#111", "width": 3}, "thickness": 0.8, "value": model["threshold"]}}))
    fig.update_layout(height=270, **{**_LAYOUT, "margin": dict(l=20, r=20, t=60, b=10)})
    return fig


def tier_bar(tiers: list[dict]) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=[t["tier"] for t in tiers], y=[t["customers"] for t in tiers],
        marker_color=[RISK_COLORS.get(t["tier"], "#888") for t in tiers],
        text=[f'{t["customers"]:,} ({t["share"]:.0%})' for t in tiers], textposition="outside"))
    fig.update_layout(title="Customers by risk tier", height=340, yaxis_title="Customers", **_LAYOUT)
    return fig


def score_hist(hist: dict, model: dict, x_title: str) -> go.Figure:
    e = hist["edges"]
    mids = [(a + b) / 2 for a, b in zip(e[:-1], e[1:])]
    fig = go.Figure(go.Bar(x=mids, y=hist["counts"], width=(e[1] - e[0]) * 0.92, marker_color="#5b8def"))
    for c, name in zip(model["risk_tiers"]["cutoffs"], model["risk_tiers"]["labels"][1:]):
        fig.add_vline(x=c, line_dash="dot", line_color=RISK_COLORS.get(name, "#888"),
                      annotation_text=f"{name} ≥ {c:g}", annotation_position="top")
    fig.add_vline(x=model["threshold"], line_color="#111", annotation_text="flag threshold",
                  annotation_position="bottom right")
    fig.update_layout(title="Score distribution", height=340, xaxis_title=x_title, yaxis_title="Customers",
                      xaxis=dict(range=[0, 1]), **_LAYOUT)
    return fig


def segment_bar(rows: list[dict], title: str) -> go.Figure:
    seg = [r["segment"] for r in rows]
    fig = go.Figure(go.Bar(x=seg, y=[r["flag_rate"] for r in rows], name="Flagged as likely to churn",
                           marker_color="#5b8def", text=[f'n={r["customers"]:,}' for r in rows], textposition="outside"))
    if rows and "actual_churn_rate" in rows[0]:
        fig.add_trace(go.Scatter(x=seg, y=[r["actual_churn_rate"] for r in rows], mode="markers+lines",
                                 name="Actual churn rate (from file)", marker=dict(size=10, color="#d64545")))
    fig.update_layout(title=title, height=380, yaxis=dict(tickformat=".0%", title="Share of customers"),
                      legend=dict(orientation="h", y=-0.2), **_LAYOUT)
    return fig


def drivers_bar(drivers: list[dict]) -> go.Figure:
    d = sorted(drivers, key=lambda x: x["importance"])
    fig = go.Figure(go.Bar(x=[x["importance"] for x in d], y=[x["label"] for x in d], orientation="h",
                           marker_color="#5b8def"))
    fig.update_layout(title="What drives churn scores across this dataset (mean |SHAP|)", height=420,
                      xaxis_title="Average influence on the score (log-odds)", **_LAYOUT)
    return fig


def contribution_bar(drivers: list[dict]) -> go.Figure:
    d = sorted(drivers, key=lambda x: abs(x["contribution"]))
    fig = go.Figure(go.Bar(
        x=[x["contribution"] for x in d], y=[x["label"] for x in d], orientation="h",
        marker_color=["#d64545" if x["contribution"] > 0 else "#2e9d6b" for x in d],
        text=[f'{x["contribution"]:+.2f}' for x in d], textposition="outside"))
    fig.update_layout(title="Why this score", height=60 + 42 * len(d), xaxis_title="← lowers risk | raises risk →",
                      **_LAYOUT)
    return fig


def confusion(c: dict) -> go.Figure:
    z = [[c["tn"], c["fp"]], [c["fn"], c["tp"]]]
    fig = go.Figure(go.Heatmap(z=z, x=["Predicted stay", "Predicted churn"], y=["Actually stayed", "Actually churned"],
                               text=[[f"{v:,}" for v in row] for row in z], texttemplate="%{text}",
                               colorscale="Blues", showscale=False))
    fig.update_layout(title="Confusion matrix", height=340, yaxis=dict(autorange="reversed"), **_LAYOUT)
    return fig


def roc(curve: dict, auc: float) -> go.Figure:
    fig = go.Figure(go.Scatter(x=curve["fpr"], y=curve["tpr"], mode="lines", name=f"ROC (AUC {auc:.3f})",
                               line=dict(color="#5b8def", width=3)))
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color="#999"))
    fig.update_layout(title="ROC curve", height=340, xaxis_title="False positive rate",
                      yaxis_title="True positive rate", legend=dict(y=0.05, x=0.55), **_LAYOUT)
    return fig
