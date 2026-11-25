---
doc_id: PK-07
title: Class Imbalance Handling
status: FACTUAL - DERIVED FROM THE CHURN NOTEBOOK
synthetic: false
source: Copy_of_Churn_Modelling_Full_Pipeline.ipynb, Section 20
created: 2026-10-07
version: 0.1
---

# Class Imbalance Handling

## 1. The problem
About 20.4% of customers churned, so most customers stayed. In the training set there are 5,571 stayed and 1,425 churned customers. A model that ignores this tends to predict "stayed" too often and misses churners.

## 2. Strategies compared
Five strategies were compared with the same model (Logistic Regression, for speed) and the same validation set, so the comparison isolates the effect of the strategy:
1. Baseline: no adjustment.
2. Class Weight: `class_weight='balanced'`, no resampling.
3. SMOTE: synthetic minority oversampling.
4. Random Oversampling: duplicate minority-class rows.
5. Random Undersampling: drop majority-class rows.
Resampling was applied to training rows only. Validation and test keep their true balance.

## 3. Results on the validation set (threshold 0.5)
| Strategy | Training rows | Val ROC-AUC | Val F1 | Val recall | Val precision |
|---|---|---|---|---|---|
| Baseline | 6,996 | 0.7812 | 0.3546 | 0.2451 | 0.6410 |
| Class Weight | 6,996 | 0.7818 | 0.5131 | 0.7206 | 0.3984 |
| SMOTE | 11,142 | 0.7793 | 0.5117 | 0.6961 | 0.4046 |
| Random Oversampling | 11,142 | 0.7825 | 0.5192 | 0.7304 | 0.4027 |
| Random Undersampling | 2,850 | 0.7808 | 0.5130 | 0.7255 | 0.3968 |

## 4. What the comparison shows
ROC-AUC barely changes between strategies (0.7793 to 0.7825). The big change is in recall and F1 at a 0.5 threshold: the baseline finds only 24.5% of churners, while every adjusted strategy finds about 70% to 73%, at the cost of lower precision. Random Oversampling had the highest validation ROC-AUC and F1 (0.7825 and 0.5192), 0.0007 above Class Weight on ROC-AUC. SMOTE had the lowest ROC-AUC.

## 5. Decision: class weighting
Class weighting (`class_weight='balanced'`) was selected for every model. The notebook's reasons: it matches the resampling methods on validation ROC-AUC while being simpler; it needs no synthetic rows; it has no risk of duplicating or discarding real customers; and it needs no extra step to apply consistently to every model. The selection was based on simplicity and near-equal ROC-AUC, not on the highest score.

## 6. How each model applies it
Logistic Regression, Random Forest and SVM use `class_weight='balanced'`. XGBoost uses `scale_pos_weight` = count of stayed / count of churned in the training set = 5,571 / 1,425, about 3.91. Naive Bayes (GaussianNB) has no class-weight option, so it relies on the validation-based decision-threshold tuning instead (best threshold 0.0149).

## 7. SMOTE in the final model
SMOTE was only tested in this comparison. It is not used in the final XGBoost model. No resampling is used in the final model.
