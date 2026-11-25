"""Aggregations over a scored dataset. Used by the dashboard endpoints AND (later) by the
chatbot's tools, so both always see the same numbers.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from ..ml.spec import ID_COLUMNS, RAW_FIELDS, TARGET_COLUMN

SEGMENT_FIELDS = ["Geography", "Gender", "AgeBand", "NumOfProducts", "IsActiveMember", "HasCrCard",
                  "CreditBand", "BalanceBand", "TenureBand"]


def _cut(s, bins, labels):
    return pd.cut(s, bins=bins, labels=labels)


def segment_series(df: pd.DataFrame, by: str) -> pd.Series:
    if by in ("Geography", "Gender"):
        return df[by]
    if by == "AgeBand":
        return _cut(df["Age"], [17, 29, 39, 49, 59, 200], ["18-29", "30-39", "40-49", "50-59", "60+"])
    if by == "NumOfProducts":
        return df["NumOfProducts"].astype(str)
    if by == "IsActiveMember":
        return df["IsActiveMember"].map({1: "Active", 0: "Inactive"})
    if by == "HasCrCard":
        return df["HasCrCard"].map({1: "Has card", 0: "No card"})
    if by == "CreditBand":
        return _cut(df["CreditScore"], [0, 499, 599, 699, 799, 900], ["<500", "500-599", "600-699", "700-799", "800+"])
    if by == "BalanceBand":
        b = df["Balance"]
        return pd.Series(np.select([b == 0, b < 50_000, b < 100_000, b < 150_000],
                                   ["Zero", "<50k", "50-100k", "100-150k"], "150k+"), index=df.index)
    if by == "TenureBand":
        return _cut(df["Tenure"], [-1, 2, 5, 8, 10], ["0-2", "3-5", "6-8", "9-10"])
    raise ValueError(f"Unknown segment '{by}'. Choose from {SEGMENT_FIELDS}.")


def segment_breakdown(results: pd.DataFrame, by: str) -> list[dict]:
    seg = segment_series(results, by)
    has_target = TARGET_COLUMN in results.columns and results[TARGET_COLUMN].notna().any()
    g = results.groupby(seg, observed=True)
    out = pd.DataFrame({
        "customers": g.size(),
        "flagged": g["will_churn"].sum(),
        "avg_score": g["churn_score"].mean(),
        "high_risk": g["risk_level"].apply(lambda s: int((s == "High").sum())),
    })
    out["flag_rate"] = out["flagged"] / out["customers"]
    if has_target:
        out["actual_churn_rate"] = g[TARGET_COLUMN].mean()
    out = out.reset_index()
    out.columns = ["segment", *out.columns[1:]]
    out["segment"] = out["segment"].astype(str)
    return out.to_dict("records")


def evaluation(results: pd.DataFrame) -> dict | None:
    if TARGET_COLUMN not in results.columns:
        return None
    lab = results[results[TARGET_COLUMN].notna()]
    if len(lab) < 20 or lab[TARGET_COLUMN].nunique() < 2:
        return None
    y = lab[TARGET_COLUMN].astype(int).to_numpy()
    score, pred = lab["churn_score"].to_numpy(), lab["will_churn"].astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y, score)
    keep = np.unique(np.linspace(0, len(fpr) - 1, 101).astype(int))
    return {
        "rows_evaluated": int(len(lab)), "actual_churn_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, score)), "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_curve": {"fpr": fpr[keep].tolist(), "tpr": tpr[keep].tolist()},
        "note": ("Computed on the uploaded rows. If this file contains customers the model was trained on, "
                 "these numbers will look better than they would on genuinely new customers."),
    }


def build_summary(results: pd.DataFrame, *, model: dict, stats: dict, stages: list, drift: list,
                  drivers: list, rejected_examples: list, out_of_range: dict) -> dict:
    n = len(results)
    tiers = model["risk_tiers"]["labels"]
    tier_counts = results["risk_level"].value_counts()
    counts, edges = np.histogram(results["churn_score"], bins=20, range=(0, 1))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {k: model[k] for k in ("name", "version", "score_type", "threshold")} | {"risk_tiers": model["risk_tiers"]},
        "kpis": {
            "customers_scored": n,
            "predicted_churners": int(results["will_churn"].sum()),
            "predicted_churn_rate": float(results["will_churn"].mean()),
            "avg_score": float(results["churn_score"].mean()),
            "high_risk": int(tier_counts.get("High", 0)),
            "rows_uploaded": stats["rows_uploaded"], "rows_rejected": stats["rows_rejected"],
            "rows_repaired": stats["rows_repaired"], "duplicates_removed": stats["duplicates_removed"],
            "outside_training_range": int(results["outside_training_range"].sum()),
        },
        "risk_tiers": [{"tier": t, "customers": int(tier_counts.get(t, 0)),
                        "share": float(tier_counts.get(t, 0) / n)} for t in tiers],
        "score_histogram": {"edges": edges.tolist(), "counts": counts.tolist()},
        "segments": {f: segment_breakdown(results, f) for f in SEGMENT_FIELDS},
        "evaluation": evaluation(results),
        "drivers": drivers,
        "quality": {"stages": stages, "issues_by_field": stats["issues_by_field"],
                    "typos_corrected": stats["typos_corrected"], "unused_columns": stats["unused_columns"],
                    "policy": stats["policy"], "rejected_examples": rejected_examples,
                    "out_of_training_range_by_field": out_of_range, "drift": drift},
    }


_SORTABLE = {"churn_score", "row_id", "Age", "Balance", "CreditScore", "EstimatedSalary", "Tenure", "NumOfProducts"}


def query_customers(results: pd.DataFrame, *, tier: str | None = None, geography: str | None = None,
                    gender: str | None = None, min_score: float | None = None, flagged_only: bool = False,
                    search: str | None = None, sort_by: str = "churn_score", ascending: bool = False,
                    limit: int = 50, offset: int = 0, search_surname: bool = True) -> tuple[int, pd.DataFrame]:
    df = results
    if tier:
        df = df[df["risk_level"].str.lower() == tier.lower()]
    if geography:
        df = df[df["Geography"].str.lower() == geography.lower()]
    if gender:
        df = df[df["Gender"].str.lower() == gender.lower()]
    if min_score is not None:
        df = df[df["churn_score"] >= min_score]
    if flagged_only:
        df = df[df["will_churn"]]
    if search:
        hay = pd.Series("", index=df.index)
        for c in (*ID_COLUMNS, "row_id"):
            if c == "Surname" and not search_surname:
                continue
            if c in df.columns:
                hay = hay + " " + df[c].astype(str)
        df = df[hay.str.contains(search, case=False, regex=False)]
    if sort_by not in _SORTABLE:
        sort_by = "churn_score"
    df = df.sort_values(sort_by, ascending=ascending, kind="stable")
    return len(df), df.iloc[offset: offset + limit]


def find_customer(results: pd.DataFrame, key: str) -> pd.Series | None:
    """Look up by CustomerId if the dataset has one, otherwise by row_id."""
    if "CustomerId" in results.columns:
        hit = results[results["CustomerId"].astype(str) == str(key)]
        if len(hit):
            return hit.iloc[0]
    if str(key).isdigit():
        hit = results[results["row_id"] == int(key)]
        if len(hit):
            return hit.iloc[0]
    return None


def frame_head(df: pd.DataFrame, n: int = 25) -> list[dict]:
    from ..utils import frame_to_records
    return frame_to_records(df.head(n))


def public_columns(results: pd.DataFrame) -> list[str]:
    lead = ["row_id", *[c for c in ID_COLUMNS if c in results.columns], *RAW_FIELDS]
    tail = [c for c in ["churn_score", "raw_score", "will_churn", "risk_level", "outside_training_range",
                        "row_status", "data_issues", TARGET_COLUMN] if c in results.columns]
    return lead + tail
