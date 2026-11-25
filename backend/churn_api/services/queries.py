"""One entry point for the read queries.

A store that can run them itself (PostgreSQL) does. Any other store (the file store) falls back to the pandas functions
in analytics.py, which are the reference behaviour. Callers never need to know which store they have.
"""
from __future__ import annotations

from . import analytics


def query_customers(store, dataset_id: str, **filters):
    native = getattr(store, "query_customers", None)
    if native is not None:
        return native(dataset_id, **filters)
    return analytics.query_customers(store.results(dataset_id), **filters)


def find_customer(store, dataset_id: str, key):
    native = getattr(store, "find_customer", None)
    if native is not None:
        return native(dataset_id, key)
    return analytics.find_customer(store.results(dataset_id), key)


def segment_breakdown(store, dataset_id: str, by: str):
    native = getattr(store, "segment_breakdown", None)
    if native is not None:
        return native(dataset_id, by)
    return analytics.segment_breakdown(store.results(dataset_id), by)