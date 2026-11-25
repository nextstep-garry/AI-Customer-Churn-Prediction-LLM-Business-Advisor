-- The customer filters compare lower(Geography), lower(Gender) and lower(risk_level); the plain indexes cannot serve them.
-- schema.py defines the same indexes (same names) for brand-new databases.
CREATE INDEX IF NOT EXISTS ix_results_geography_lower ON dataset_results (dataset_id, lower("Geography"));
CREATE INDEX IF NOT EXISTS ix_results_gender_lower ON dataset_results (dataset_id, lower("Gender"));
CREATE INDEX IF NOT EXISTS ix_results_tier_lower ON dataset_results (dataset_id, lower(risk_level));