"""Read-side queries that PostgreSQL runs itself, so a page view or a chat question no longer loads a whole dataset.

PostgresDatasetStore inherits these three methods. The file store has no such methods, so services/queries.py falls back
to the pandas functions in services/analytics.py, which are the reference behaviour. Both must give identical answers:
tests/test_pg_queries.py runs both on the same data and compares.

Rules copied from analytics.py (the SQL spells each one out, because PostgreSQL's defaults differ from pandas'):
  - text groups sort by code point (COLLATE "C"); band groups sort in band order
  - rows without a segment drop out of a breakdown
  - NULL sorts last, and ties keep row_id order (pandas' stable sort)
  - search is a case-insensitive substring of " <CustomerId> <Surname> <row_id>" (a missing value counts as empty)
"""
from __future__ import annotations

import threading
from collections import OrderedDict

import pandas as pd
import sqlalchemy as sa

from ..ml.spec import ID_COLUMNS
from . import schema as s
from .datasets import DatasetNotFound

T = s.dataset_results
RESULT_COLUMNS = [c.name for c in T.columns if c.name != "dataset_id"]
OPTIONAL_COLUMNS = ("CustomerId", "Surname", "raw_score", "Exited")      # absent from a result frame when the upload lacked them
SORTABLE = {"churn_score", "row_id", "Age", "Balance", "CreditScore", "EstimatedSalary", "Tenure", "NumOfProducts"}
SEGMENTS = ["Geography", "Gender", "AgeBand", "NumOfProducts", "IsActiveMember", "HasCrCard",
            "CreditBand", "BalanceBand", "TenureBand"]
MAX_ROW_ID = 2_147_483_647                                              # row_id is a 32-bit integer in the database
_FACTS_LOCK = threading.Lock()
_FACTS_KEEP = 8


def _text(expr):
    return sa.cast(expr, sa.Text)


def _band(column, edges, labels):
    """pandas.cut(column, edges, labels): right-closed intervals (e0, e1], (e1, e2], ... Returns (label, order)."""
    steps = [(sa.and_(column > lo, column <= hi), i) for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))]
    label = _text(sa.case(*[(cond, sa.literal(labels[i])) for cond, i in steps]))
    order = sa.cast(sa.case(*[(cond, sa.literal(i)) for cond, i in steps]), sa.Integer)
    return label, order


def segment_expressions(by: str):
    """(segment label, sort key). A sort key of None means: sort the labels themselves by code point, as pandas does."""
    if by in ("Geography", "Gender"):
        return T.c[by], None
    if by == "AgeBand":
        return _band(T.c.Age, [17, 29, 39, 49, 59, 200], ["18-29", "30-39", "40-49", "50-59", "60+"])
    if by == "NumOfProducts":
        return _text(T.c.NumOfProducts), None
    if by == "IsActiveMember":
        return _text(sa.case((T.c.IsActiveMember == 1, "Active"), (T.c.IsActiveMember == 0, "Inactive"))), None
    if by == "HasCrCard":
        return _text(sa.case((T.c.HasCrCard == 1, "Has card"), (T.c.HasCrCard == 0, "No card"))), None
    if by == "CreditBand":
        return _band(T.c.CreditScore, [0, 499, 599, 699, 799, 900], ["<500", "500-599", "600-699", "700-799", "800+"])
    if by == "BalanceBand":
        b = T.c.Balance          # a missing balance falls through to "150k+", exactly as numpy.select does in analytics.py
        return _text(sa.case((b == 0, "Zero"), (b < 50_000, "<50k"), (b < 100_000, "50-100k"),
                             (b < 150_000, "100-150k"), else_="150k+")), None
    if by == "TenureBand":
        return _band(T.c.Tenure, [-1, 2, 5, 8, 10], ["0-2", "3-5", "6-8", "9-10"])
    raise ValueError(f"Unknown segment '{by}'. Choose from {SEGMENTS}.")


class PgQueries:
    """Mixed into PostgresDatasetStore. Needs self.engine and self._check()."""

    # ------------------------------------------------------------------ what this dataset looks like (cached)
    def _facts(self, dataset_id: str) -> dict:
        """Dataset-wide facts that decide how a page of rows must look to match the full pandas frame:
        which optional columns exist, and whether every CustomerId is numeric. Cached per dataset version."""
        self._check(dataset_id)
        with self.engine.connect() as conn:
            head = conn.execute(sa.select(s.datasets.c.updated_at, s.datasets.c.has_target)
                                .where(s.datasets.c.id == dataset_id)).first()
            if head is None:
                raise DatasetNotFound(dataset_id)
            cache = self.__dict__.setdefault("_facts_cache", OrderedDict())
            key = (dataset_id, head.updated_at)
            with _FACTS_LOCK:
                if key in cache:
                    cache.move_to_end(key)
                    return cache[key]
            counts = conn.execute(sa.select(*[sa.func.count(T.c[n]).label(n) for n in OPTIONAL_COLUMNS])
                                  .where(T.c.dataset_id == dataset_id)).mappings().one()
            ids = [r[0] for r in conn.execute(sa.select(T.c.CustomerId)
                                              .where(T.c.dataset_id == dataset_id, T.c.CustomerId.is_not(None)))]
        numeric = pd.to_numeric(pd.Series(ids, dtype="object"), errors="coerce") if ids else None
        facts = {
            "present": {n for n in OPTIONAL_COLUMNS if counts[n] > 0 or (n == "Exited" and head.has_target)},
            "has_exited_values": counts["Exited"] > 0,
            "ids_numeric": bool(ids) and int(numeric.notna().sum()) == len(ids),
        }
        with _FACTS_LOCK:
            cache[key] = facts
            while len(cache) > _FACTS_KEEP:
                cache.popitem(last=False)
        return facts

    @staticmethod
    def _columns(facts: dict) -> list[str]:
        return [c for c in RESULT_COLUMNS if c not in OPTIONAL_COLUMNS or c in facts["present"]]

    @staticmethod
    def _page(rows, facts: dict) -> pd.DataFrame:
        df = pd.DataFrame([dict(r) for r in rows], columns=PgQueries._columns(facts))
        if "CustomerId" in df.columns and facts["ids_numeric"]:
            df["CustomerId"] = pd.to_numeric(df["CustomerId"], errors="coerce")
        return df

    # ------------------------------------------------------------------ customers
    def query_customers(self, dataset_id: str, *, tier=None, geography=None, gender=None, min_score=None,
                        flagged_only=False, search=None, sort_by="churn_score", ascending=False,
                        limit=50, offset=0, search_surname=True) -> tuple[int, pd.DataFrame]:
        facts = self._facts(dataset_id)
        where = [T.c.dataset_id == dataset_id]
        if tier:
            where.append(sa.func.lower(T.c.risk_level) == tier.lower())
        if geography:
            where.append(sa.func.lower(T.c.Geography) == geography.lower())
        if gender:
            where.append(sa.func.lower(T.c.Gender) == gender.lower())
        if min_score is not None:
            where.append(T.c.churn_score >= min_score)
        if flagged_only:
            where.append(T.c.will_churn.is_(True))
        if search:
            hay = None
            for name in (*ID_COLUMNS, "row_id"):
                if name == "row_id" or (name in facts["present"] and (name != "Surname" or search_surname)):
                    piece = _text(sa.literal(" ")).concat(sa.func.coalesce(_text(T.c[name]), _text(sa.literal(""))))
                    hay = piece if hay is None else hay.concat(piece)
            escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")    # a literal match, not wildcards
            where.append(hay.ilike(f"%{escaped}%"))
        sort_col = T.c[sort_by if sort_by in SORTABLE else "churn_score"]
        order = [(sort_col.asc() if ascending else sort_col.desc()).nulls_last(), T.c.row_id.asc()]
        cols = [T.c[n] for n in self._columns(facts)]
        with self.engine.connect() as conn:
            total = conn.execute(sa.select(sa.func.count()).select_from(T).where(*where)).scalar()
            rows = conn.execute(sa.select(*cols).where(*where).order_by(*order)
                                .limit(int(limit)).offset(int(offset))).mappings().all()
        return int(total), self._page(rows, facts)

    def find_customer(self, dataset_id: str, key):
        """By CustomerId if the dataset has one, otherwise (or if nothing matches) by row_id, like analytics.find_customer."""
        facts = self._facts(dataset_id)
        cols = [T.c[n] for n in self._columns(facts)]
        key = str(key)

        def first(condition):
            with self.engine.connect() as conn:
                rows = conn.execute(sa.select(*cols).where(T.c.dataset_id == dataset_id, condition)
                                    .order_by(T.c.row_id).limit(1)).mappings().all()
            return self._page(rows, facts).iloc[0] if rows else None

        if "CustomerId" in facts["present"]:
            hit = first(T.c.CustomerId == key)
            if hit is not None:
                return hit
        if key.isdigit():
            try:
                number = int(key)
            except ValueError:                    # exotic Unicode digits such as "²"
                return None
            if number <= MAX_ROW_ID:
                return first(T.c.row_id == number)
        return None

    # ------------------------------------------------------------------ segments
    def segment_breakdown(self, dataset_id: str, by: str) -> list[dict]:
        facts = self._facts(dataset_id)
        label, order = segment_expressions(by)
        inner = (sa.select(label.label("segment"), (label if order is None else order).label("ord"),
                           T.c.will_churn, T.c.churn_score, T.c.risk_level, T.c.Exited)
                 .where(T.c.dataset_id == dataset_id).subquery())
        columns = [inner.c.segment,
                   sa.func.count().label("customers"),
                   sa.func.sum(sa.case((inner.c.will_churn.is_(True), 1), else_=0)).label("flagged"),
                   sa.cast(sa.func.avg(inner.c.churn_score), sa.Float).label("avg_score"),
                   sa.func.sum(sa.case((inner.c.risk_level == "High", 1), else_=0)).label("high_risk")]
        if facts["has_exited_values"]:
            columns.append(sa.cast(sa.func.avg(sa.cast(inner.c.Exited, sa.Float)), sa.Float).label("actual_churn_rate"))
        sort_key = sa.collate(inner.c.ord, "C") if order is None else inner.c.ord
        stmt = (sa.select(*columns).where(inner.c.segment.is_not(None))
                .group_by(inner.c.segment, inner.c.ord).order_by(sort_key))
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        out = []
        for r in rows:
            customers, flagged = int(r["customers"]), int(r["flagged"])
            rec = {"segment": str(r["segment"]), "customers": customers, "flagged": flagged,
                   "avg_score": float(r["avg_score"]), "high_risk": int(r["high_risk"]),
                   "flag_rate": flagged / customers}
            if facts["has_exited_values"]:
                rec["actual_churn_rate"] = None if r["actual_churn_rate"] is None else float(r["actual_churn_rate"])
            out.append(rec)
        return out