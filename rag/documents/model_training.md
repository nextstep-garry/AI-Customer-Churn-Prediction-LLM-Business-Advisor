---
doc_id: PK-08
title: Model Training and the Five Models
status: FACTUAL - DERIVED FROM THE CHURN NOTEBOOK
synthetic: false
source: Copy_of_Churn_Modelling_Full_Pipeline.ipynb, Sections 21 and 22
created: 2026-10-07
version: 0.1
---

# Model Training

## 1. Five models compared
The churn project trained and compared five models on the same 13 features: Logistic Regression, Random Forest, Naive Bayes (GaussianNB), SVM (RBF kernel) and XGBoost. All use class weighting (`class_weight='balanced'`, or `scale_pos_weight` for XGBoost). Naive Bayes has no class-weight option. Every model went through the same validation-based threshold tuning and was scored on train, validation and test for transparency, but only validation scores were used for selection.

## 2. Logistic Regression
Settings: max_iter 1000, class_weight balanced, random_state 42, no hyperparameter search. At threshold 0.5 it scored train ROC-AUC 0.7785, validation ROC-AUC 0.7818 and test ROC-AUC 0.7646. It is the simplest and most interpretable baseline model.
Coefficients on the scaled features, largest first: IsActiveMember -0.934, Age +0.847, Geography_germany +0.793, Gender -0.602, Balance +0.202, HasCrCard -0.114, CreditScore -0.077, EstimatedSalary +0.065, NumOfProducts -0.053, Geography_spain -0.037, Tenure -0.034, Zero_Balance_Flag +0.018, Balance_Salary_Ratio +0.001. Positive means higher churn risk. With Gender coded female = 0 and male = 1, the negative coefficient means male customers have lower predicted risk.

## 3. Random Forest
RandomForestClassifier with class_weight balanced, tuned by RandomizedSearchCV (20 iterations). Best parameters: n_estimators 500, max_depth 10, min_samples_leaf 5, min_samples_split 5, max_features log2. Train ROC-AUC 0.9511 against validation 0.8629 shows a large train-validation gap.

## 4. Naive Bayes
GaussianNB tuned with GridSearchCV over var_smoothing. Best var_smoothing 2.68e-05. Its default 0.5 threshold gave a validation F1 of 0.009, so it relies heavily on threshold tuning.

## 5. SVM
SVC with RBF kernel, class_weight balanced, probability=True, tuned by RandomizedSearchCV (10 iterations). Best parameters: C 0.1, gamma 0.1.

## 6. XGBoost
XGBClassifier tuned by RandomizedSearchCV (20 iterations), then refit with early stopping. Best parameters: subsample 0.8, reg_lambda 5.0, reg_alpha 0, min_child_weight 5, max_depth 4, learning_rate 0.03, colsample_bytree 0.6. Refit with n_estimators up to 1000 and early_stopping_rounds 30 on the validation set; the best iteration was 62. Train ROC-AUC 0.8800, validation 0.8664, test 0.8605.

## 7. Per-model results (tuned decision threshold)
| Model | Threshold | Train AUC | Val AUC | Test AUC | Test precision | Test recall | Test F1 | Test accuracy |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.4951 | 0.7785 | 0.7818 | 0.7646 | 0.3893 | 0.7002 | 0.5004 | 0.7154 |
| Random Forest | 0.4802 | 0.9511 | 0.8629 | 0.8549 | 0.5708 | 0.6830 | 0.6219 | 0.8309 |
| Naive Bayes | 0.0149 | 0.7867 | 0.7879 | 0.7656 | 0.4717 | 0.5725 | 0.5172 | 0.7824 |
| SVM (RBF) | 0.2971 | 0.8519 | 0.8470 | 0.8327 | 0.5009 | 0.6683 | 0.5726 | 0.7969 |
| XGBoost | 0.5000 | 0.8800 | 0.8664 | 0.8605 | 0.5100 | 0.7494 | 0.6070 | 0.8024 |
The two tree ensembles (XGBoost and Random Forest) clearly beat the other three on ROC-AUC.
