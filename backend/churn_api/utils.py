"""JSON helpers: numpy/pandas values in, strictly valid JSON (no NaN/Infinity) out."""
from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
import pandas as pd
from flask.json.provider import DefaultJSONProvider


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


def frame_to_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of plain dicts; NaN becomes null."""
    return to_jsonable(df.astype(object).where(df.notna(), None).to_dict("records"))


class ChurnJSONProvider(DefaultJSONProvider):
    sort_keys = False

    def dumps(self, obj, **kwargs):
        return super().dumps(to_jsonable(obj), **kwargs)
