"""Personal data in API output. Surnames are stored, but every response hides them unless the request asks for them
(`?include_surname=true`) and the operator allows it (ALLOW_SURNAME_ACCESS). The assistant's tools never return them."""
from __future__ import annotations

import pandas as pd

SENSITIVE_COLUMNS = ("Surname",)


def strip_frame(df: pd.DataFrame) -> pd.DataFrame:
    """A copy of `df` without the sensitive columns (the original is untouched)."""
    return df.drop(columns=[c for c in SENSITIVE_COLUMNS if c in df.columns])


def strip_row(row: pd.Series) -> pd.Series:
    """A copy of one customer row without the sensitive fields."""
    return row.drop(labels=[c for c in SENSITIVE_COLUMNS if c in row.index])



def strip_json(value):
    """A copy of a JSON-like structure with every sensitive key removed, at any depth (used for stored summaries)."""
    if isinstance(value, dict):
        return {k: strip_json(v) for k, v in value.items() if k not in SENSITIVE_COLUMNS}
    if isinstance(value, list):
        return [strip_json(v) for v in value]
    return value