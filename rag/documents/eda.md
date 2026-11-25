---
doc_id: PK-03
title: Exploratory Data Analysis, Outliers and Scaling Exploration
status: FACTUAL - DERIVED FROM THE CHURN NOTEBOOK
synthetic: false
source: Copy_of_Churn_Modelling_Full_Pipeline.ipynb, Sections 9, 12 and 13
created: 2026-10-07
version: 0.1
---

# Exploratory Data Analysis

## 1. What EDA covered
EDA in the churn project ran on the cleaned, still human-readable data (before encoding). It produced: count plots of Exited, IsActiveMember and HasCrCard; churn rate by Gender, IsActiveMember and Geography; a correlation heatmap with the target; and a heatmap of correlations among the numeric features. The plots are images in the notebook. Numeric churn rates per category: Not available in the current project artifacts (the notebook shows them only as bar charts).

## 2. Class balance
Exited is imbalanced. From the split support counts, the churn rate is about 20.4%: train 1,425 churned of 6,996, validation 204 of 1,000, test 407 of 1,999.

## 3. Correlation of numeric features with Exited
| Feature | Correlation with Exited |
|---|---|
| Age | 0.2849 |
| Zero_Balance_Flag | -0.1209 |
| Balance | 0.1176 |
| NumOfProducts | -0.0470 |
| CreditScore | -0.0277 |
| Balance_Salary_Ratio | 0.0255 |
| Tenure | -0.0139 |
| EstimatedSalary | 0.0125 |
Age has the strongest linear relationship with churn. Values near 0 mean little linear relationship; NumOfProducts has a weak linear correlation but a strong non-linear effect that shows up in SHAP (see shap_analysis.md).

## 4. Multicollinearity check
A correlation matrix among the six numeric columns was plotted to look for near-duplicate features. The numeric matrix values are not printed in the notebook: Not available in the current project artifacts.

## 5. Outlier detection (exploratory, full data)
IQR rule (bounds Q1 - 1.5 IQR and Q3 + 1.5 IQR) and Z-score rule (absolute Z above 3) on the cleaned data:
| Column | IQR bounds | IQR outliers | Z-score outliers |
|---|---|---|---|
| CreditScore | 384.5 to 916.5 | 16 | 8 |
| Age | 14 to 62 | 357 | 132 |
| Tenure | -3 to 13 | 0 | 0 |
| Balance | -191,348.17 to 318,913.62 | 0 | 0 |
| NumOfProducts | -0.5 to 3.5 | 60 | 60 |
| EstimatedSalary | -95,802.83 to 296,018.71 | 0 | 0 |
This exploration only decided whether capping was needed. The capping that feeds the model is done after the split using training-only bounds (see leakage_safe_processing.md).

## 6. Skewness
CreditScore -0.07, Age 1.01, Tenure 0.01, Balance -0.14, NumOfProducts 0.75, EstimatedSalary 0.00. Age is the most right-skewed column.

## 7. Scaling exploration (visual comparison only)
Min-Max scaling and standard scaling were both applied to every numeric column to compare them visually, creating `Scaled_<column>(MinMax)` and `Scaled_<column>(Standard)` columns. These Scaled_* columns are for comparison only and are NOT fed to any model. The scaling that feeds the models is a separate StandardScaler fitted on training data only.
