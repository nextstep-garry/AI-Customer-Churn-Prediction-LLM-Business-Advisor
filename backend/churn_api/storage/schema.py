"""PostgreSQL table definitions (SQLAlchemy Core). The single source of truth for the database layout.

    datasets          one row per upload: metadata, row counts and the precomputed dashboard summary (JSONB)
    dataset_raw       the original CSV bytes, kept apart so listing datasets never drags megabytes along
    dataset_results   one row per scored customer; the column names are exactly the DataFrame's, so the swap is mechanical
    dataset_rejected  rows set aside by validation; values may be invalid (text in a number column), so they are JSONB
    predictions       audit trail of single-customer predictions
    schema_version    which layout version the database has

Deleting a dataset deletes everything that belongs to it (ON DELETE CASCADE), in one transaction.
When the layout changes, bump SCHEMA_VERSION and add a migration (Alembic is the next step up from this).
"""
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

SCHEMA_VERSION = 3     # 3: added Surname column to dataset_results

metadata = sa.MetaData()
_now = sa.func.now()

datasets = sa.Table(
    "datasets", metadata,
    sa.Column("id", sa.String(32), primary_key=True),                 # server-generated uuid hex, as in the file store
    sa.Column("name", sa.Text, nullable=False),
    sa.Column("original_filename", sa.Text, nullable=False),
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("policy", sa.Text, nullable=False),
    sa.Column("has_target", sa.Boolean, nullable=False),
    sa.Column("model_name", sa.Text, nullable=False),
    sa.Column("model_version", sa.Text, nullable=False),
    sa.Column("score_type", sa.Text, nullable=False),
    sa.Column("rows_uploaded", sa.Integer, nullable=False),
    sa.Column("rows_scored", sa.Integer, nullable=False),
    sa.Column("rows_rejected", sa.Integer, nullable=False),
    sa.Column("rows_repaired", sa.Integer, nullable=False),
    sa.Column("duplicates_removed", sa.Integer, nullable=False),
    sa.Column("summary", JSONB, nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=_now),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=_now),
    sa.Index("ix_datasets_created_at", "created_at"),
)

dataset_raw = sa.Table(
    "dataset_raw", metadata,
    sa.Column("dataset_id", sa.String(32), sa.ForeignKey("datasets.id", ondelete="CASCADE"), primary_key=True),
    sa.Column("csv", sa.LargeBinary, nullable=False),
)

dataset_results = sa.Table(
    "dataset_results", metadata,
    sa.Column("dataset_id", sa.String(32), sa.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False),
    sa.Column("row_id", sa.Integer, nullable=False),
    sa.Column("CustomerId", sa.Text),                           # optional in uploads
    sa.Column("Surname", sa.Text),                                    # personal data: see the hardening step
    sa.Column("CreditScore", sa.Integer),
    sa.Column("Geography", sa.Text),
    sa.Column("Gender", sa.Text),
    sa.Column("Age", sa.Integer),
    sa.Column("Tenure", sa.Integer),
    sa.Column("Balance", sa.Float),
    sa.Column("NumOfProducts", sa.Integer),
    sa.Column("HasCrCard", sa.Integer),
    sa.Column("IsActiveMember", sa.Integer),
    sa.Column("EstimatedSalary", sa.Float),
    sa.Column("churn_score", sa.Float, nullable=False),
    sa.Column("raw_score", sa.Float),                                 # only for calibrated models: the score before calibration
    sa.Column("will_churn", sa.Boolean, nullable=False),
    sa.Column("risk_level", sa.Text, nullable=False),
    sa.Column("outside_training_range", sa.Boolean, nullable=False),
    sa.Column("row_status", sa.Text, nullable=False),
    sa.Column("data_issues", sa.Text, nullable=False, server_default=""),
    sa.Column("Exited", sa.Integer),                                  # only when the upload had the target column
    sa.PrimaryKeyConstraint("dataset_id", "row_id"),
    sa.Index("ix_results_score", "dataset_id", sa.text("churn_score DESC")),   # "top risk" and the default sort
    sa.Index("ix_results_tier", "dataset_id", "risk_level"),
    sa.Index("ix_results_geography", "dataset_id", "Geography"),
    sa.Index("ix_results_customer", "dataset_id", "CustomerId"),
)


# The customer filters compare lower(column), which the plain indexes above cannot serve. The SQL migration
# 0003_case_insensitive_filter_indexes.sql adds the same indexes, with the same names, to databases created earlier.
sa.Index("ix_results_geography_lower", dataset_results.c.dataset_id, sa.func.lower(dataset_results.c.Geography))
sa.Index("ix_results_gender_lower", dataset_results.c.dataset_id, sa.func.lower(dataset_results.c.Gender))
sa.Index("ix_results_tier_lower", dataset_results.c.dataset_id, sa.func.lower(dataset_results.c.risk_level))
# sa.Index("ix_results_surname_lower", dataset_results.c.dataset_id, sa.func.lower(dataset_results.c.Surname))

dataset_rejected = sa.Table(
    "dataset_rejected", metadata,
    sa.Column("dataset_id", sa.String(32), sa.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False),
    sa.Column("row_id", sa.Integer, nullable=False),
    sa.Column("data_issues", sa.Text, nullable=False, server_default=""),
    sa.Column("record", sa.JSON, nullable=False),                       # the original (possibly invalid) values
    sa.PrimaryKeyConstraint("dataset_id", "row_id"),
)

predictions = sa.Table(
    "predictions", metadata,
    sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
    sa.Column("ts", sa.DateTime(timezone=True), nullable=False, server_default=_now),
    sa.Column("request_id", sa.Text),
    sa.Column("model_name", sa.Text),
    sa.Column("model_version", sa.Text),
    sa.Column("input", sa.JSON, nullable=False),
    sa.Column("score", sa.Float, nullable=False),
    sa.Column("score_type", sa.Text),
    sa.Column("risk_level", sa.Text),
    sa.Column("will_churn", sa.Boolean, nullable=False),
)

schema_version = sa.Table(
    "schema_version", metadata,
    sa.Column("version", sa.Integer, primary_key=True),
    sa.Column("applied_at", sa.DateTime(timezone=True), nullable=False, server_default=_now),
)