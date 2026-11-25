"""Uploaded dataset -> validated frame. Encodes the notebook's cleaning rules (audit 8.3d):

  * header tolerance      "credit_score", "Credit Score", "CREDITSCORE" all map to CreditScore
  * duplicate rows        exact duplicates (ignoring the RowNumber index column) are dropped
  * typo'd categories     franc / spainn / germnny are corrected by fuzzy match; 'usa' is not
  * invalid values        sentinels (-99999.99, 999999999.99 ...), out-of-range, non-numeric,
                          non-integer counts, missing
  * policy                "reject": invalid rows are set aside and reported (default, safest)
                          "repair": invalid cells are replaced by the column mode/median of the
                                    valid rows in the same file -- how the notebook treated its
                                    raw training data. Every repaired row is flagged.

Nothing here scores or transforms features; it only decides which rows are fit to score.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .exceptions import DatasetError
from .spec import (CATEGORICAL, CATEGORY_ALIASES, EXCLUSIVE_MIN, ID_COLUMNS, IGNORED_COLUMNS, INT_FIELDS, MISSING_TOKENS,
                   MODE_FIELDS, NUMERIC_FIELDS, RANGES, RAW_FIELDS, TARGET_COLUMN)

_key = lambda name: re.sub(r"[^a-z0-9]", "", str(name).lower())  # noqa: E731
_CANON = {_key(n): n for n in [*RAW_FIELDS, TARGET_COLUMN, *ID_COLUMNS, *IGNORED_COLUMNS]}


@dataclass
class CleanResult:
    valid: pd.DataFrame           # rows fit to score (canonical columns, proper dtypes)
    rejected: pd.DataFrame        # rows set aside, original values + reasons
    stages: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def canonicalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={c: _CANON[_key(c)] for c in df.columns if _key(c) in _CANON})
    dup = out.columns[out.columns.duplicated()].tolist()
    if dup:
        raise DatasetError(f"Column names collide after normalising headers: {dup}")
    return out


def _blank(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.strip() == "")


def clean_dataset(raw: pd.DataFrame, policy: str = "reject") -> CleanResult:
    if policy not in {"reject", "repair"}:
        raise DatasetError("policy must be 'reject' or 'repair'.")
    if raw.empty:
        raise DatasetError("The file has no data rows.")

    n_uploaded = len(raw)
    df = canonicalise_columns(raw)
    missing_cols = [c for c in RAW_FIELDS if c not in df.columns]
    if missing_cols:
        raise DatasetError(f"Missing required column(s): {missing_cols}. Columns found: {list(df.columns)}")
    extra = [c for c in df.columns if c not in RAW_FIELDS and c not in ID_COLUMNS
             and c not in IGNORED_COLUMNS and c != TARGET_COLUMN]

    df = df.copy()
    df.insert(0, "row_id", np.arange(1, n_uploaded + 1))
    stages = [{"stage": "read", "label": "Read file", "rows_in": n_uploaded, "rows_out": n_uploaded,
               "detail": f"{len(raw.columns)} columns" + (f", {len(extra)} unused ignored" if extra else "")}]

    # ---- 1. exact duplicates -------------------------------------------------------------
    subset = [c for c in df.columns if c != "row_id" and c not in IGNORED_COLUMNS]
    dup = df.duplicated(subset=subset, keep="first")
    n_dup = int(dup.sum())
    df = df.loc[~dup].reset_index(drop=True)
    stages.append({"stage": "dedupe", "label": "Remove duplicate rows", "rows_in": n_uploaded,
                   "rows_out": len(df), "detail": f"{n_dup} removed"})

    problems: dict[str, pd.Series] = {}
    original = df[RAW_FIELDS].copy()  # untouched values, for the rejected-rows report

    def flag(fld: str, mask: pd.Series, text: str) -> None:
        s = problems.setdefault(fld, pd.Series("", index=df.index, dtype=object))
        s.loc[mask & (s == "")] = f"{fld}: {text}"

    # ---- 2. categories: normalise + fix typos ----------------------------------------------
    typos = 0
    for fld, cats in CATEGORICAL.items():
        norm = df[fld].astype(str).str.strip().str.lower()
        blank = _blank(df[fld]) | norm.isin(MISSING_TOKENS)
        norm = norm.where(~blank)
        canon = {c.lower(): c for c in cats}
        lookup = {**canon, **{k.lower(): v for k, v in CATEGORY_ALIASES.get(fld, {}).items()}}
        exact = norm.map(lookup)
        fix = {}
        for v in norm[exact.isna() & ~blank].dropna().unique():
            hit = difflib.get_close_matches(v, list(lookup), n=1, cutoff=0.8)
            if hit:
                fix[v] = lookup[hit[0]]
        fuzzy = norm.map(fix) if fix else pd.Series(np.nan, index=df.index, dtype=object)
        values = exact.where(exact.notna(), fuzzy)
        typos += int((values.notna() & ~norm.isin(list(canon))).sum())   # typos + abbreviations resolved
        df[fld] = values.astype(object)
        flag(fld, blank, "missing")
        flag(fld, values.isna() & ~blank, "unrecognised value")

    # ---- 3. numeric fields: parse, range, integrality ------------------------------------------
    for fld in NUMERIC_FIELDS:
        blank = _blank(df[fld])
        num = pd.to_numeric(df[fld], errors="coerce").astype(float)
        flag(fld, blank, "missing")
        flag(fld, num.isna() & ~blank, "not a number")
        finite = np.isfinite(num)
        flag(fld, num.notna() & ~finite, "not finite")
        num = num.where(finite)
        lo, hi = RANGES[fld]
        below = (num <= lo) if fld in EXCLUSIVE_MIN else (num < lo)
        flag(fld, below | (num > hi), f"outside valid range ({'>' if fld in EXCLUSIVE_MIN else ''}{lo:,.10g} to {hi:,.10g})")
        if fld in INT_FIELDS:
            flag(fld, num.notna() & (num % 1 != 0), "must be a whole number")
        df[fld] = num

    bad = pd.DataFrame({f: (s != "") for f, s in problems.items()}) if problems else pd.DataFrame(index=df.index)
    row_bad = bad.any(axis=1) if not bad.empty else pd.Series(False, index=df.index)
    reasons = pd.Series("", index=df.index, dtype=object)
    if row_bad.any():
        joined = pd.concat([problems[f] for f in problems], axis=1)
        reasons.loc[row_bad] = joined.loc[row_bad].apply(lambda r: "; ".join(x for x in r if x), axis=1)
    issues_by_field = {f: int(bad[f].sum()) for f in bad.columns if bad[f].sum()} if not bad.empty else {}

    # ---- 4. policy ------------------------------------------------------------------------
    df["row_status"] = "ok"
    df["data_issues"] = ""
    n_repaired = 0
    if policy == "repair" and row_bad.any():
        unrepairable = pd.Series(False, index=df.index)
        for fld in bad.columns:
            cells = bad[fld]
            if not cells.any():
                continue
            good = df.loc[~cells, fld].dropna()
            if good.empty:
                unrepairable |= cells
                continue
            fill = good.mode().iloc[0] if fld in MODE_FIELDS else good.median()
            if fld in INT_FIELDS:
                fill = int(round(float(fill)))
            df.loc[cells, fld] = fill
        n_repaired = int((row_bad & ~unrepairable).sum())
        df.loc[row_bad & ~unrepairable, "row_status"] = "repaired"
        df.loc[row_bad & ~unrepairable, "data_issues"] = "repaired: " + reasons[row_bad & ~unrepairable]
        reject_mask = unrepairable
    else:
        reject_mask = row_bad

    rejected = pd.concat([df.loc[reject_mask, ["row_id", *[c for c in ID_COLUMNS if c in df.columns]]],
                          original.loc[reject_mask]], axis=1)
    rejected["data_issues"] = reasons[reject_mask]
    valid = df.loc[~reject_mask].reset_index(drop=True)
    for fld in INT_FIELDS:
        valid[fld] = valid[fld].astype("int64")
    if TARGET_COLUMN in valid.columns:
        y = pd.to_numeric(valid[TARGET_COLUMN], errors="coerce")
        valid[TARGET_COLUMN] = y.where(y.isin([0, 1]))

    stages.insert(2, {"stage": "standardise", "label": "Standardise categories", "rows_in": len(df),
                      "rows_out": len(df), "detail": f"{typos} typos / abbreviations corrected"})
    n_rej = int(reject_mask.sum())
    detail = f"{n_repaired} repaired, {n_rej} rejected" if policy == "repair" else f"{n_rej} rejected"
    stages.append({"stage": "validate", "label": "Validate values", "rows_in": len(df),
                   "rows_out": len(valid), "detail": detail})
    stats = {
        "rows_uploaded": n_uploaded, "duplicates_removed": n_dup, "typos_corrected": typos,
        "rows_rejected": int(reject_mask.sum()), "rows_repaired": n_repaired, "rows_valid": len(valid),
        "issues_by_field": issues_by_field, "policy": policy,
        "target_present": TARGET_COLUMN in valid.columns and bool(valid[TARGET_COLUMN].notna().any()),
        "unused_columns": extra,
    }
    return CleanResult(valid=valid, rejected=rejected.reset_index(drop=True), stages=stages, stats=stats)
