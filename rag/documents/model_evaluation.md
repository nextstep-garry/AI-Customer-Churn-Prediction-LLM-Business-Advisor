---
doc_id: PK-10
title: Model Comparison, Selection and Final Evaluation
status: FACTUAL - DERIVED FROM THE CHURN NOTEBOOK
synthetic: false
source: Copy_of_Churn_Modelling_Full_Pipeline.ipynb, Sections 22, 25, 26, 27 and 28
created: 2026-10-07
version: 0.1
---

# Model Comparison, Selection and Final Evaluation

## 1. How the final model was selected
The final churn model is chosen by validation ROC-AUC, not test ROC-AUC. Using test ROC-AUC would make the test set part of model selection, so it would stop being an unbiased final check. The selection order is: cross-validation ROC-AUC inside each tuning step, then validation ROC-AUC to pick the best model, then validation F1 to tune that model's threshold, then one final evaluation on the test set. In code: `BEST_MODEL_NAME = comparison['val_roc_auc'].idxmax()`.

## 2. Validation ROC-AUC ranking
| Rank | Model | Validation ROC-AUC |
|---|---|---|
| 1 | XGBoost | 0.8664 |
| 2 | Random Forest | 0.8629 |
| 3 | SVM (RBF) | 0.8470 |
| 4 | Naive Bayes | 0.7879 |
| 5 | Logistic Regression | 0.7818 |
The winner is XGBoost. Its margin over Random Forest on validation ROC-AUC is 0.0035.

## 3. Validation metrics for all models
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.720 | 0.3984 | 0.7304 | 0.5156 | 0.7818 |
| Random Forest | 0.839 | 0.5850 | 0.7255 | 0.6477 | 0.8629 |
| Naive Bayes | 0.776 | 0.4603 | 0.5686 | 0.5088 | 0.7879 |
| SVM (RBF) | 0.807 | 0.5206 | 0.6814 | 0.5902 | 0.8470 |
| XGBoost | 0.804 | 0.5129 | 0.7794 | 0.6187 | 0.8664 |

## 4. Final test-set result for XGBoost
The final test ROC-AUC of the selected XGBoost model was 0.8605, with precision 0.5100 and recall 0.7494 at threshold 0.500.
The test set (1,999 customers, 407 churned) was evaluated for the chosen model at threshold 0.500:
| Metric | Value |
|---|---|
| ROC-AUC | 0.8605 |
| Precision | 0.5100 |
| Recall | 0.7494 |
| F1 | 0.6070 |
| Accuracy | 0.8024 |
Per class on the test set: Stayed precision 0.927, recall 0.816; Churned precision 0.510, recall 0.749, F1 0.607. In plain terms, the model finds about 75% of customers who actually churn, and about half of the customers it flags as churners really do churn.

## 5. Test results for the other models (reported for transparency, not used for selection)
Random Forest had higher test precision (0.5708), F1 (0.6219) and accuracy (0.8309) than XGBoost, and XGBoost had higher test recall (0.7494 against 0.6830) and higher ROC-AUC (0.8605 against 0.8549). The ordering of test ROC-AUC matches the ordering of validation ROC-AUC: XGBoost 0.8605, Random Forest 0.8549, SVM 0.8327, Naive Bayes 0.7656, Logistic Regression 0.7646.

## 6. Overfitting check
Train, validation and test ROC-AUC for XGBoost are 0.8800, 0.8664 and 0.8605, a small gap. Random Forest shows 0.9511, 0.8629 and 0.8549, a much larger train-versus-validation gap.

## 7. Saved model
The notebook saves a single pickle file `final_model.pkl` containing: the fitted model, model_name, the decision threshold (0.500), feature_names (the 13 features in order), the fitted StandardScaler and the winsorisation bounds. The deployed churn app uses exported artifacts instead: `xgb_model.json` (XGBoost in its native format) and `preprocessing.pkl` (encoders, winsorisation bounds, scaler), described by `model_manifest.json` (see api_documentation.md). The app documentation names the notebook's original files `churn_model.pkl` and `model_metadata.pkl`, which differs from the `final_model.pkl` name seen in the notebook's save cell.

## 8. Inference smoke test
The saved pickle was loaded back and used to score one row taken from the test set. The row was already winsorised and scaled. The model returned a predicted churn score of 0.4735, below the 0.500 threshold, so the predicted class was Stayed, and the actual class was Stayed. This test checks that the artifact loads and predicts; it did not start from a raw, unscaled customer record.

## 9. Conclusion items in the notebook
Best model XGBoost; selection metric validation ROC-AUC; validation ROC-AUC 0.8664; test ROC-AUC 0.8605; test precision 0.5100; test recall 0.7494; test F1 0.6070; threshold 0.500; top SHAP drivers NumOfProducts, Age, IsActiveMember, Geography_germany, Gender.

## 10. Why XGBoost was selected
XGBoost was selected as the final churn model because it had the highest validation ROC-AUC of the five models: 0.8664, against 0.8629 for Random Forest, 0.8470 for SVM, 0.7879 for Naive Bayes and 0.7818 for Logistic Regression. The selection rule was validation ROC-AUC, not test ROC-AUC. XGBoost also had the highest test recall (0.7494) and a small train-to-validation gap (0.8800 to 0.8664). Random Forest had higher test precision and F1, but those were not the selection criteria.

## 11. Final test recall and precision
The final XGBoost model's test recall was 0.7494: it found about 75% of the customers who actually churned. The final XGBoost model's test precision was 0.5100: about half of the customers it flagged as churners really did churn. The final XGBoost model's test F1 was 0.6070 and its accuracy was 0.8024, at threshold 0.500.
