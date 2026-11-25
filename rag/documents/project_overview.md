---
doc_id: PK-01
title: Churn Project Overview
status: FACTUAL - DERIVED FROM THE CHURN NOTEBOOK
synthetic: false
source: Copy_of_Churn_Modelling_Full_Pipeline.ipynb (all sections), project owner's description
created: 2026-10-07
version: 0.1
---

# Churn Project Overview

## 1. What the project is
The Customer Churn Analysis and Prediction project predicts whether a bank customer will leave the bank. The target column is `Exited` (1 = the customer churned, 0 = the customer stayed). The raw file is `Churn_Modelling_dirty_raw.csv`, a bank-customer dataset that deliberately contains dirty values such as wrong data types, placeholder codes, typos and duplicates. The work was done in a Google Colab notebook with data stored on Google Drive.

## 2. Two separate functions: ML and RAG
The churn project has two independent functions.
- The machine-learning model predicts churn for a customer (customer features in, churn score out).
- The RAG assistant answers questions about the project (for example "Why was class weighting used?"). It reads project documents, not customer records.
The RAG assistant is never part of preprocessing, feature selection, training, evaluation or prediction.

## 3. Pipeline stages
| Stage | Notebook sections | Knowledge document |
|---|---|---|
| Target, missing values, duplicates, invalid data, categorical cleaning | 3-7 | data_preparation.md |
| Feature creation and encoding | 8, 11 | feature_engineering.md |
| EDA, outlier and scaling exploration | 9, 12, 13 | eda.md |
| Filter and wrapper feature selection | 10, 14, 18 | feature_selection.md |
| Split, winsorisation, scaling, saving data | 15, 16, 17, 19 | leakage_safe_processing.md |
| Class imbalance comparison | 20 | class_imbalance.md |
| Five models | 21 | model_training.md |
| Tuning and threshold selection | 21 | hyperparameter_tuning.md |
| Comparison, selection, final test, saved model | 22, 25, 26, 27, 28 | model_evaluation.md |
| SHAP and PCA | 23, 24 | shap_analysis.md |
| Backend API | not in the notebook | api_documentation.md |

## 4. Dataset size at each step
- Raw file: 10,015 rows and 15 columns.
- After removing 15 exact duplicates: 10,000 rows.
- After dropping 5 rows with a missing target: 9,995 rows.
- Split: train 6,996 rows, validation 1,000 rows, test 1,999 rows (about 70% / 10% / 20%).
- The model uses 13 features.

## 5. Final outcome
The selected model is XGBoost. It was chosen by validation ROC-AUC (0.8664). On the test set, evaluated once after selection, it scored ROC-AUC 0.8605, precision 0.5100, recall 0.7494, F1 0.6070 and accuracy 0.8024 at a decision threshold of 0.50. Details are in model_evaluation.md.

## 6. Libraries seen in the notebook
pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, imbalanced-learn (SMOTE, random over- and under-sampling), xgboost and shap. The deployed model was built with scikit-learn 1.6.1, xgboost 3.4.1 and joblib 1.6.0 (model_manifest.json). Versions of the other notebook libraries: Not available in the current project artifacts.

## 7. Data tiers
The notebook uses three folders: `raw/` (untouched file), `preprocessed/` (cleaned and encoded, 9,995 rows by 18 columns) and `final/` (train, validation and test files after winsorisation and scaling).
