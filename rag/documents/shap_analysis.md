---
doc_id: PK-11
title: SHAP Explainability and PCA
status: FACTUAL - DERIVED FROM THE CHURN NOTEBOOK
synthetic: false
source: Copy_of_Churn_Modelling_Full_Pipeline.ipynb, Sections 23 and 24
created: 2026-10-07
version: 0.1
---

# SHAP Explainability and PCA

## 1. Why SHAP was used
SHAP (SHapley Additive exPlanations) attributes each prediction to the features that pushed it up or down. It is more informative than a single feature-importance number because it shows direction and size per customer. In the churn project SHAP was used to explain the tuned XGBoost model, using the fast exact TreeExplainer for tree models. The notebook calls SHAP the most actionable output for the business. SHAP values describe what the model used, not what causes churn.

## 2. How it was run
`shap.TreeExplainer(xgb_best)` was applied to the test features X_test. The notebook produced a SHAP summary plot and a bar plot, and ranked features by mean absolute SHAP value.

## 3. Strongest churn drivers (mean absolute SHAP value)
| Rank | Feature | Mean absolute SHAP |
|---|---|---|
| 1 | NumOfProducts | 0.3960 |
| 2 | Age | 0.3946 |
| 3 | IsActiveMember | 0.1922 |
| 4 | Geography_germany | 0.1214 |
| 5 | Gender | 0.1145 |
| 6 | Balance | 0.0751 |
| 7 | Balance_Salary_Ratio | 0.0349 |
| 8 | Zero_Balance_Flag | 0.0349 |
| 9 | CreditScore | 0.0217 |
| 10 | EstimatedSalary | 0.0190 |
Tenure, HasCrCard and Geography_spain are not among the top ten. NumOfProducts and Age are the two dominant drivers, about twice as influential as the third.

## 4. What PCA was used for
PCA (Principal Component Analysis) is dimensionality reduction. It builds new axes that are combinations of all features instead of keeping or dropping the original columns. In the churn project PCA was shown as a separate analysis to satisfy the technique requirement and as a variance sanity check, and to project the data into 2D.

## 5. Was PCA used as feature selection?
No. PCA was not used for feature selection and did not replace the model features. The models were kept on the original 13 interpretable features so that SHAP can attribute importance to real customer attributes instead of opaque components. PCA was fitted on X_train only.

## 6. PCA results
- The notebook printed: "Components needed for 90% variance: 1 / 13 original features".
- The 2D projection showed the two classes (stayed and churned) heavily overlapping rather than forming clean clusters.
- The commentary text in the notebook says PCA "doesn't compress much" and takes most of the original components to reach 90% variance. That statement does not match the printed result of 1 out of 13. The notebook does not explain the difference.
