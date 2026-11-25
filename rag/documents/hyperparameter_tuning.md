---
doc_id: PK-09
title: Hyperparameter Tuning and Threshold Tuning
status: FACTUAL - DERIVED FROM THE CHURN NOTEBOOK
synthetic: false
source: Copy_of_Churn_Modelling_Full_Pipeline.ipynb, Section 21 (tuning helpers and model cells)
created: 2026-10-07
version: 0.1
---

# Hyperparameter Tuning and Threshold Tuning

## 1. Cross-validation setup
All searches use StratifiedKFold with 5 splits, shuffle enabled and random_state 42, scored by ROC-AUC, on training data only. The seed is 42 for every search.

## 2. Search methods per model
| Model | Method | Iterations or grid | Best CV ROC-AUC |
|---|---|---|---|
| Logistic Regression | none | fixed settings | not tuned |
| Random Forest | RandomizedSearchCV | 20 iterations | 0.8633 |
| Naive Bayes | GridSearchCV | var_smoothing, 15 values from 1e-11 to 1e-1 (logspace) | 0.7824 |
| SVM | RandomizedSearchCV | 10 iterations | 0.8321 |
| XGBoost | RandomizedSearchCV | 20 iterations | 0.8672 |

## 3. Search spaces
- Random Forest: n_estimators [200, 300, 500]; max_depth [4, 6, 8, 10, None]; min_samples_leaf [1, 3, 5, 10]; min_samples_split [2, 5, 10]; max_features [sqrt, log2, 0.5].
- SVM: C from 10^-2 to 10^1 (10 values); gamma from 10^-4 to 10^-1 (10 values); kernel rbf.
- XGBoost: max_depth [3, 4, 5, 6]; learning_rate [0.01, 0.03, 0.05, 0.1]; subsample [0.6, 0.8, 1.0]; colsample_bytree [0.6, 0.8, 1.0]; reg_alpha [0, 0.1, 1.0]; reg_lambda [1.0, 2.0, 5.0]; min_child_weight [1, 3, 5].

## 4. XGBoost early stopping
After the random search, XGBoost is refit with the best parameters, n_estimators 1000, early_stopping_rounds 30 and eval_metric AUC, using the validation set as the early-stopping set. Early stopping with 30 rounds selected a best iteration of 62. The search itself used n_estimators 300 and scale_pos_weight of about 3.91.

## 5. Decision threshold tuning
For every one of the five models the churn decision threshold was tuned on the validation set: 100 evenly spaced thresholds from 0.01 to 0.5 were swept, and the threshold with the highest validation F1 was chosen. Results:
| Model | Best threshold | Val F1 at best | Val F1 at 0.5 |
|---|---|---|---|
| Logistic Regression | 0.495 | 0.516 | 0.513 |
| Random Forest | 0.480 | 0.648 | 0.631 |
| Naive Bayes | 0.015 | 0.509 | 0.009 |
| SVM | 0.297 | 0.590 | 0.530 |
| XGBoost | 0.500 | 0.619 | 0.619 |
Naive Bayes needed a very low threshold (0.015): at the default 0.5 its validation F1 was only 0.009. The notebook does not state why. For XGBoost the best threshold equals the top of the swept range (0.5), so thresholds above 0.5 were not evaluated.

## 6. Which data each step used
Cross-validation scoring: training folds only. XGBoost early stopping, threshold tuning and model selection: validation set. Test set: never used for tuning. Because the validation set is used for early stopping, threshold tuning and model selection together, validation scores can be slightly optimistic compared with test scores (XGBoost validation ROC-AUC 0.8664 against test 0.8605).
