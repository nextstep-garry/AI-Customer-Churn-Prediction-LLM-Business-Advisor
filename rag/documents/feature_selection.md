---
doc_id: PK-05
title: Feature Selection Results
status: FACTUAL - DERIVED FROM THE CHURN NOTEBOOK
synthetic: false
source: Copy_of_Churn_Modelling_Full_Pipeline.ipynb, Sections 10, 14 and 18
created: 2026-10-07
version: 0.1
---

# Feature Selection Results

## 1. Methods used
The churn project used three filter methods on the full cleaned data (Chi-square for categorical and binary features, Variance Threshold, Mutual Information for numeric features) and two wrapper methods on training data only (Forward and Backward Sequential Feature Selection). SHAP is used for explainability and PCA for dimensionality reduction; neither is a feature-selection step in this project (see shap_analysis.md).

## 2. Chi-square test (Geography, Gender, HasCrCard, IsActiveMember)
A feature is statistically related to churn when p < 0.05.
| Feature | Chi2 | p-value | Significant |
|---|---|---|---|
| Geography | 295.65 | 0.0000 | Yes |
| Gender | 113.22 | 0.0000 | Yes |
| HasCrCard | 0.52 | 0.4708 | No |
| IsActiveMember | 242.26 | 0.0000 | Yes |
HasCrCard is the only feature whose chi-square result was not significant. It was still kept, because the notebook treats the chi-square table as one input to the decision rather than an automatic drop rule, and with about 10,000 rows significance alone does not guarantee predictive usefulness.

## 3. Variance Threshold (threshold 0.01 on standardised data)
All eight numeric and engineered features had a standardised variance of 1.0 and were above the threshold. Low-variance candidates for dropping: none. Because each feature is standardised before the test, every variance equals 1.0 by construction.

## 4. Mutual Information (numeric features vs Exited)
| Feature | Mutual information |
|---|---|
| Age | 0.0693 |
| NumOfProducts | 0.0674 |
| Balance | 0.0111 |
| Balance_Salary_Ratio | 0.0091 |
| Zero_Balance_Flag | 0.0074 |
| CreditScore | 0.0064 |
| EstimatedSalary | 0.0028 |
| Tenure | 0.0000 |
Age and NumOfProducts carry far more information about churn than the others. Tenure scored 0.0 but was not dropped.

## 5. Decision: no features removed
No filter method led to a feature being dropped. In the notebook `features_to_drop = []`. No columns were dropped by the filter methods: all 13 model features were kept. CreditScore_Tier and the Scaled_* columns were never part of the 13 model features. The model matrix excludes only CreditScore_Tier and the Scaled_* columns (visualisation-only) plus the identifier columns. The final model has 13 features.

## 6. Wrapper selection (training data only)
Forward and Backward Sequential Feature Selection used LogisticRegression (max_iter 1000, class_weight balanced), n_features_to_select='auto', tol 0.001, 5-fold cross-validation and ROC-AUC scoring, fitted on X_train and y_train only.
- Forward selection kept 5 of 13: Gender, Age, Balance, IsActiveMember, Geography_germany.
- Backward selection kept 12 of 13: everything except Tenure.
- Both directions agreed on: Age, Balance, Gender, Geography_germany, IsActiveMember.
Forward selection did not pick NumOfProducts, which is the top feature by SHAP for XGBoost; the wrapper used a linear model.

## 7. Decision on the wrapper results
The full 13-feature set was kept for the model comparison. Reasons given in the notebook: no filter method flagged a feature as uninformative, and swapping in the forward or backward subset would make Random Forest, XGBoost and SHAP results harder to compare like for like. The disagreement between forward and backward selection is noted as the first place to check if the model is revisited.
