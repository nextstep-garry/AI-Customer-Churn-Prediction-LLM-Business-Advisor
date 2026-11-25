"""ChurnPipeline: the whole ML pipeline behind one object.

    single customer  ->  predict_records([...])            (validated dicts from the API)
    whole dataset    ->  run_dataset(raw_dataframe, ...)   (clean -> features -> score -> summarise)

The pipeline holds no state except its ModelBundle, so replacing the model is
`ChurnPipeline(ModelBundle.load(new_dir))` -- the chatbot, dashboards and API all sit on top of this.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..services import analytics
from .bundle import ModelBundle
from .cleaning import clean_dataset
from .drift import mean_shift
from .exceptions import DatasetError
from .explain import explain_row, global_drivers
from .features import build_feature_frame
from .spec import ID_COLUMNS, RAW_FIELDS, TARGET_COLUMN

log = logging.getLogger(__name__)


@dataclass
class PipelineRun:
    results: pd.DataFrame
    rejected: pd.DataFrame
    summary: dict
    stages: list[dict]
    stats: dict


class ChurnPipeline:
    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle

    # ------------------------------------------------------------------ helpers
    def _score(self, clean: pd.DataFrame):
        b = self.bundle
        X = build_feature_frame(clean, b)
        raw = b.predict_raw(X)
        score = b.calibrate(raw)
        return X, raw, score

    def _out_of_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Boolean frame: which numeric cells fall outside the range seen in training."""
        flags = {}
        for col, (lo, hi) in self.bundle.training_ranges.items():
            if col in df.columns:
                flags[col] = (df[col] < lo) | (df[col] > hi)
        return pd.DataFrame(flags, index=df.index)

    def _label(self, will_churn: bool) -> str:
        return "Likely to churn" if will_churn else "Likely to stay"

    # ------------------------------------------------------------------ single / small batch
    def predict_records(self, records: list[dict], explain: bool = False, top_k: int = 8) -> list[dict]:
        b = self.bundle
        df = pd.DataFrame(records)[RAW_FIELDS]
        X, raw, score = self._score(df)
        oor = self._out_of_range(df)
        levels = b.risk_levels(score)
        out = []
        for i, rec in enumerate(records):
            will = bool(score[i] >= b.threshold)
            item = {
                "churn_score": round(float(score[i]), 4), "score_type": b.score_type,
                "will_churn": will, "prediction_label": self._label(will),
                "risk_level": str(levels[i]), "threshold_used": b.threshold,
                "warnings": [
                    f"{c} = {rec[c]:,.0f} is outside the range seen in training "
                    f"({b.training_ranges[c][0]:,.0f} to {b.training_ranges[c][1]:,.0f}); "
                    "the score may be less reliable."
                    for c in oor.columns if oor.iloc[i][c]
                ],
            }
            if explain:
                item["explanation"] = explain_row(b, X.iloc[[i]], rec, top_k=top_k)
            out.append(item)
        return out

    # ------------------------------------------------------------------ whole dataset
    def run_dataset(self, raw_df: pd.DataFrame, policy: str = "reject") -> PipelineRun:
        t0, b = time.perf_counter(), self.bundle
        cleaned = clean_dataset(raw_df, policy)
        valid = cleaned.valid
        if valid.empty:
            raise DatasetError(
                f"None of the {cleaned.stats['rows_uploaded']} rows passed validation "
                f"(issues by field: {cleaned.stats['issues_by_field']}). Check the file against GET /v1/schema.")

        X, raw, score = self._score(valid)
        results = valid.copy()
        results["churn_score"] = np.round(score, 6)
        if b.score_type == "calibrated":
            results["raw_score"] = np.round(raw, 6)
        results["will_churn"] = score >= b.threshold
        results["risk_level"] = b.risk_levels(score)
        oor = self._out_of_range(valid)
        results["outside_training_range"] = oor.any(axis=1) if not oor.empty else False
        results = results[analytics.public_columns(results)]

        stages = list(cleaned.stages)
        stages.append({"stage": "features", "label": "Feature engineering, encoding, scaling",
                       "rows_in": len(valid), "rows_out": len(valid),
                       "detail": f"{len(b.feature_names)} model features"})
        stages.append({"stage": "score", "label": f"Score with {b.model_name} v{b.model_version}",
                       "rows_in": len(valid), "rows_out": len(results),
                       "detail": f"{int(results['will_churn'].sum())} flagged at threshold {b.threshold:g}"})

        rejected_examples = analytics.frame_head(cleaned.rejected) if len(cleaned.rejected) else []
        summary = analytics.build_summary(
            results, model=b.info(), stats=cleaned.stats, stages=stages, drift=mean_shift(valid, b.prep),
            drivers=global_drivers(b, X), rejected_examples=rejected_examples,
            out_of_range={c: int(oor[c].sum()) for c in oor.columns if oor[c].any()})
        stages.append({"stage": "summarise", "label": "Build dashboard summary", "rows_in": len(results),
                       "rows_out": len(results), "detail": "KPIs, segments, drivers, drift"})
        summary["quality"]["stages"] = stages
        log.info("dataset_scored", extra={"rows_in": cleaned.stats["rows_uploaded"], "rows_scored": len(results),
                                          "rows_rejected": cleaned.stats["rows_rejected"], "policy": policy,
                                          "seconds": round(time.perf_counter() - t0, 3)})
        return PipelineRun(results=results, rejected=cleaned.rejected, summary=summary, stages=stages, stats=cleaned.stats)

    def explain_customer(self, row: pd.Series, top_k: int = 8) -> dict:
        rec = {f: (row[f].item() if hasattr(row[f], "item") else row[f]) for f in RAW_FIELDS}
        X, _, _ = self._score(pd.DataFrame([rec]))
        return explain_row(self.bundle, X, rec, top_k=top_k)
