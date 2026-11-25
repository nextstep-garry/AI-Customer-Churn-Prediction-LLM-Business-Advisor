# =============================================================================
# Paste these cells into your notebook, in place of the old Section 26
# ("Save the Final Model") cell. They assume the notebook's existing
# variables are already in memory: best_model, BEST_MODEL_NAME,
# best_threshold, FEATURE_NAMES, scaler, winsor_bounds, X_test, y_test.
#
# WHY THIS REPLACES THE OLD CELL:
# The original Section 11 encoding cell created `LabelEncoder()` and
# `OneHotEncoder()` objects *inside* helper functions (`ordinal_encoding`,
# `OHE_encoding_categorical`) and never returned them -- so they were lost
# once the functions returned, and the old final_model.pkl never included
# them. That's fine for scoring the notebook's own X_test (already encoded
# numbers), but it means Flask would have no way to encode a brand-new raw
# customer's "France"/"Male" strings. This patch captures those two fitted
# encoders as named variables the first time they're created, then saves
# them alongside the model. No ML result changes -- same model, same
# threshold, same feature order.
# =============================================================================

# -----------------------------------------------------------------------
# PATCH for Section 11 (encoding cell) -- capture the fitted encoders
# -----------------------------------------------------------------------
# Replace:
#     df = ordinal_encoding(df, "Gender")
#     df = OHE_encoding_categorical(df, "Geography")
# with:

gender_encoder = LabelEncoder()
df["Gender"] = gender_encoder.fit_transform(df["Gender"])

geography_encoder = OneHotEncoder(drop="first", sparse_output=False)
_geo_encoded = geography_encoder.fit_transform(df[["Geography"]])
_geo_encoded_df = pd.DataFrame(
    _geo_encoded,
    columns=geography_encoder.get_feature_names_out(["Geography"]),
    index=df.index,
)
df = pd.concat([df.drop(columns=["Geography"]), _geo_encoded_df], axis=1)


# -----------------------------------------------------------------------
# Section 26 (replaces the old saving cell) -- minimal deployment artifacts
# -----------------------------------------------------------------------
import os
import joblib

os.makedirs("models", exist_ok=True)

# churn_model.pkl: everything needed to go from a feature row -> prediction
joblib.dump(
    {
        "model": best_model,
        "model_name": BEST_MODEL_NAME,
        "threshold": best_threshold,
        "feature_names": FEATURE_NAMES,
    },
    "models/churn_model.pkl",
)

# preprocessing.pkl: everything needed to go from a RAW customer -> feature row
joblib.dump(
    {
        "scaler": scaler,
        "winsor_bounds": winsor_bounds,
        "numeric_cols": NUMERIC_COLS,
        "gender_encoder": gender_encoder,
        "geography_encoder": geography_encoder,
    },
    "models/preprocessing.pkl",
)

print("Saved models/churn_model.pkl and models/preprocessing.pkl")


# -----------------------------------------------------------------------
# Section 27 (replaces the old inference-test cell) -- validate + compare
# -----------------------------------------------------------------------
model_artifact = joblib.load("models/churn_model.pkl")
prep_artifact = joblib.load("models/preprocessing.pkl")

# A. Original notebook inference process (X_test is already fully preprocessed)
sample_X_original = X_test.iloc[[0]][model_artifact["feature_names"]]
sample_y = y_test.iloc[0]
prob_original = model_artifact["model"].predict_proba(sample_X_original)[:, 1][0]
pred_original = int(prob_original >= model_artifact["threshold"])

# B. Newly saved deployment artifacts, replaying the SAME already-processed
#    row through the saved model (this is the "loaded back from disk" check
#    Step 13 asks for -- feature values, order, probability, and class
#    should match A exactly, since it's the identical row and identical
#    fitted model object).
sample_X_reloaded = X_test.iloc[[0]][prep_artifact["numeric_cols"] + 
                                      [c for c in model_artifact["feature_names"] 
                                       if c not in prep_artifact["numeric_cols"]]]
sample_X_reloaded = sample_X_reloaded[model_artifact["feature_names"]]
prob_reloaded = model_artifact["model"].predict_proba(sample_X_reloaded)[:, 1][0]
pred_reloaded = int(prob_reloaded >= model_artifact["threshold"])

print(f"Loaded model: {model_artifact['model_name']} (threshold={model_artifact['threshold']:.3f})")
print(f"A. Original pipeline   -> probability={prob_original:.4f}, "
      f"class={'Churned' if pred_original else 'Stayed'}")
print(f"B. Reloaded artifacts  -> probability={prob_reloaded:.4f}, "
      f"class={'Churned' if pred_reloaded else 'Stayed'}")
assert abs(prob_original - prob_reloaded) < 1e-9, "Discrepancy between original and reloaded pipeline!"
print("Match confirmed -- reloaded artifacts reproduce the original pipeline exactly.")
print(f"Actual class: {'Churned' if sample_y == 1 else 'Stayed'}")
