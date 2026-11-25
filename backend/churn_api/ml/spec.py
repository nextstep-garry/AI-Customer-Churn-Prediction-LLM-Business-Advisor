"""Raw-data contract: the single source of truth for field names, ranges and categories.

Ranges come from the notebook's own cleaning rules (audit report section 8.3d).
Both the single-record JSON schema (pydantic) and the dataset cleaner (pandas) read
from here, so the API and the batch path can never disagree.
"""
from __future__ import annotations

RAW_FIELDS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
TARGET_COLUMN = "Exited"                 # optional in uploads; enables the evaluation section
ID_COLUMNS = ("CustomerId", "Surname")   # optional pass-through columns
IGNORED_COLUMNS = ("RowNumber",)         # row index in the raw file; never a feature

INT_FIELDS = ("CreditScore", "Age", "Tenure", "NumOfProducts", "HasCrCard", "IsActiveMember")
FLOAT_FIELDS = ("Balance", "EstimatedSalary")
NUMERIC_FIELDS = INT_FIELDS + FLOAT_FIELDS

CATEGORICAL = {
    "Geography": ("France", "Germany", "Spain"),
    "Gender": ("Female", "Male"),
}

# Accepted abbreviations (found in the real raw file: 'f', 'm'), and tokens that mean "no value".
CATEGORY_ALIASES = {"Gender": {"m": "Male", "f": "Female"}}
MISSING_TOKENS = frozenset({"nan", "na", "n/a", "null", "none", "unknown", "???", "-"})

# Inclusive (min, max). EstimatedSalary must be strictly greater than its minimum.
RANGES = {
    "CreditScore": (350, 850),
    "Age": (18, 92),
    "Tenure": (0, 10),
    "Balance": (0.0, 1e7),
    "NumOfProducts": (1, 4),
    "HasCrCard": (0, 1),
    "IsActiveMember": (0, 1),
    "EstimatedSalary": (0.0, 1e7),
}
EXCLUSIVE_MIN = frozenset({"EstimatedSalary"})

# With policy="repair", bad cells are replaced the way the notebook repaired its raw data:
# mode for discrete/categorical fields, median otherwise (statistics from the uploaded file).
MODE_FIELDS = frozenset({"Geography", "Gender", "NumOfProducts", "HasCrCard", "IsActiveMember"})

LABELS = {
    "CreditScore": "Credit score", "Geography": "Country", "Gender": "Gender", "Age": "Age",
    "Tenure": "Tenure (years)", "Balance": "Account balance", "NumOfProducts": "Number of products",
    "HasCrCard": "Has credit card", "IsActiveMember": "Active member", "EstimatedSalary": "Estimated salary",
}


def field_specs() -> list[dict]:
    """Machine-readable description of the input contract (served at GET /v1/schema)."""
    out = []
    for name in RAW_FIELDS:
        if name in CATEGORICAL:
            out.append({"name": name, "label": LABELS[name], "type": "categorical",
                        "choices": list(CATEGORICAL[name])})
        else:
            lo, hi = RANGES[name]
            out.append({"name": name, "label": LABELS[name],
                        "type": "integer" if name in INT_FIELDS else "number",
                        "min": lo, "max": hi, "exclusive_min": name in EXCLUSIVE_MIN})
    return out
