---
doc_id: PK-04
title: Feature Engineering and Encoding
status: FACTUAL - DERIVED FROM THE CHURN NOTEBOOK
synthetic: false
source: Copy_of_Churn_Modelling_Full_Pipeline.ipynb, Sections 8 and 11
created: 2026-10-07
version: 0.1
---

# Feature Engineering and Encoding

## 1. Features created for the model
Two engineered features are used by the churn model: Zero_Balance_Flag and Balance_Salary_Ratio. They are computed on the cleaned data, before any scaling.

## 2. Zero_Balance_Flag
Zero_Balance_Flag is 1 when the customer's Balance equals 0, otherwise 0. The idea, noted in the notebook as general banking knowledge rather than a verified source, is that an empty account can mean the customer already moved money elsewhere. In the data, 3,602 of 9,995 customers have a zero balance. The observed churn rate was 13.88% for zero-balance customers and 24.03% for customers with a non-zero balance. So in this dataset zero-balance customers churned less often, not more.

## 3. Balance_Salary_Ratio
Balance_Salary_Ratio is Balance divided by EstimatedSalary. There is no "+1" in the denominator. The notebook explains that EstimatedSalary was already cleaned to be positive (minimum 11.58), so dividing directly is safe, and an arbitrary +1 would have distorted the ratio for every row. Example rows: Balance 0 gives ratio 0.0; Balance 83,807.86 with salary 112,542.58 gives 0.7447.

## 4. CreditScore_Tier (visualisation only)
CreditScore_Tier bins CreditScore using bin edges 300, 580, 670, 740, 800, 900 with labels Poor, Fair, Good, Very Good, Excellent. It was created for an exploratory plot only. It is NOT a model feature. The notebook keeps the continuous CreditScore instead, because tree and linear models can learn their own thresholds without losing information to fixed bins. CreditScore_Tier and all Scaled_* columns are explicitly excluded from the model matrix.

## 5. Transformations defined but not applied
The notebook defines reusable helpers for log, power, Box-Cox and Yeo-Johnson transforms and for equal-width, equal-frequency and KBins binning. Only the credit-score tier binning was applied to the data. None of the transforms changed the model features.

## 6. Encoding
Encoding runs after EDA and the chi-square test, because those steps need the original text labels.
- Gender uses LabelEncoder. LabelEncoder assigns codes in alphabetical order, so female = 0 and male = 1.
- Geography uses one-hot encoding with `drop='first'`. France is the baseline category, so the model columns are Geography_germany and Geography_spain.
After encoding and dropping Surname, the preprocessed data has 9,995 rows and 18 columns.

## 7. Final 13 model features
CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Zero_Balance_Flag, Balance_Salary_Ratio, Geography_germany, Geography_spain.
