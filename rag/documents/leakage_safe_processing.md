---
doc_id: PK-06
title: Leakage-Safe Processing (Split, Winsorisation, Scaling)
status: FACTUAL - DERIVED FROM THE CHURN NOTEBOOK
synthetic: false
source: Copy_of_Churn_Modelling_Full_Pipeline.ipynb, Sections 15, 16, 17, 18 and 19
created: 2026-10-07
version: 0.1
---

# Leakage-Safe Processing

## 1. Train, validation and test split
Data leakage in the churn project is controlled by splitting first and fitting data-dependent steps on training data only. The split has two stages, both stratified on Exited with random seed 42:
1. 20% is held out as the test set (1,999 rows).
2. From the remaining 80%, 12.5% is held out as validation (1,000 rows); the rest is training (6,996 rows).
This gives about 70% train, 10% validation and 20% test. Class counts: train 5,571 stayed / 1,425 churned; validation 796 / 204; test 1,592 / 407.
The test set is treated as a final untouched exam: it is never used to choose models, tune hyperparameters or pick thresholds.

## 2. Winsorisation bounds from training data only
Winsorisation (capping outliers) uses the IQR rule: lower = Q1 - 1.5 x IQR and upper = Q3 + 1.5 x IQR. The bounds are computed from X_train only and then applied unchanged to validation and test. Bounds learned from the training set:
| Column | Lower | Upper | Rows capped in train |
|---|---|---|---|
| CreditScore | 383.12 | 916.12 | 12 |
| Age | 14.00 | 62.00 | 247 |
| Tenure | -4.50 | 15.50 | 0 |
| Balance | -190,930.27 | 318,217.12 | 0 |
| NumOfProducts | -0.50 | 3.50 | 48 |
| EstimatedSalary | -96,971.17 | 295,371.07 | 0 |
Customers older than 62 are capped at 62, and 4-product customers are capped at 3.5.

## 3. StandardScaler fitted on training data only
A StandardScaler is fitted on X_train and then used to transform validation and test. It is applied after winsorisation. It scales six columns only: CreditScore, Age, Tenure, Balance, NumOfProducts and EstimatedSalary. Gender, HasCrCard, IsActiveMember, Zero_Balance_Flag, Balance_Salary_Ratio, Geography_germany and Geography_spain are not winsorised or scaled by this step.

## 4. Other training-only steps
- Wrapper feature selection (Forward and Backward) was fitted on X_train and y_train only.
- Resampling for the imbalance comparison was applied to training rows only; validation and test keep their true class balance.
- PCA was fitted on X_train only.
- Hyperparameter search used 5-fold cross-validation inside the training set. XGBoost early stopping and every model's decision threshold used the validation set.

## 5. Steps that used the full dataset before the split
For accuracy, these steps ran on all cleaned rows before the split: imputation values (medians and modes), the filter feature-selection tests (Chi-square, Variance Threshold, Mutual Information), the exploratory outlier, skewness and scaling analysis, and the encoders (LabelEncoder for Gender, OneHotEncoder for Geography). None of these changed the model's feature list.

## 6. Saved datasets
After winsorisation and scaling, three files were saved in the `final` folder: churn_train.csv (6,996 rows, 14 columns), churn_val.csv (1,000 rows, 14 columns) and churn_test.csv (1,999 rows, 14 columns). Each has the 13 features plus Exited.
