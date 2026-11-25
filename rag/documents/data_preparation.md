---
doc_id: PK-02
title: Data Preparation (Cleaning)
status: FACTUAL - DERIVED FROM THE CHURN NOTEBOOK
synthetic: false
source: Copy_of_Churn_Modelling_Full_Pipeline.ipynb, Sections 3-7
created: 2026-10-07
version: 0.1
---

# Data Preparation (Cleaning)

## 1. Target and feature groups
In the churn project the target is `Exited`. Features are grouped by type:
- Numerical: CreditScore, Age, Tenure, Balance, NumOfProducts, EstimatedSalary.
- Categorical: Geography, Gender.
- Binary: HasCrCard, IsActiveMember.
Identifier columns (`Unnamed: 0`, `RowNumber`, `CustomerId`) and `Surname` are not model features.

## 2. Hidden missing values and the missing-value strategy
The raw file contains hidden missing values: wrong data types, out-of-range numbers and placeholder codes. In the churn project every such value is first converted to NaN, then imputed. In summary: Age is imputed with the mean (rounded to 39); Tenure, CreditScore, Balance and EstimatedSalary are imputed with the median; NumOfProducts, HasCrCard, IsActiveMember, Geography and Gender are imputed with the mode.
The method per column is:
| Column | Invalid rule (converted to NaN) | Imputation | Value used |
|---|---|---|---|
| Age | 0 or below, or above 92 (raw min -5, raw max 999) | Mean, hard-coded as rounded value | 39 (mean 38.92, median 37.0) |
| Tenure | outside 0-10 | Median | 5.0 |
| NumOfProducts | codes -1, 0, 80, 999, then anything outside 1-4 | Mode | 1.0 |
| CreditScore | below 350 or above 850 | Median | 652.0 (mean 650.55) |
| Balance | placeholders -99999.99 and 999999999.99 | Median | 97,258.33 |
| EstimatedSalary | 0 or below, or placeholder 99999999.99 | Median | 100,200.40 |
| HasCrCard | value 2 (5 rows) | Mode | 1.0 |
| IsActiveMember | value 2 (1 row) | Mode | 1.0 |
| Geography | unknown tokens and "USA" | Mode | france |
| Gender | unknown tokens | Mode | male |
All numeric and binary columns were first converted with `pd.to_numeric(errors='coerce')`, so any non-numeric junk became NaN. After coercion the NaN counts were: CreditScore 18, Age 18, Tenure 14, Balance 20, NumOfProducts 1, EstimatedSalary 19.

## 3. Placeholder values treated as missing
The placeholder and invalid codes found in the churn data were:
- Age: -5 and 999 (and anything outside 1-92).
- NumOfProducts: -1, 0, 80, 999.
- Balance: -99999.99 (4 rows) and 999999999.99 (8 rows).
- EstimatedSalary: 0 (1 row) and 99999999.99.
- HasCrCard: 2 (5 rows). IsActiveMember: 2 (1 row).
- Geography: "nan", "na", "<na>", "unknown", "???" and "USA".
- Gender: "nan", "na", "u", "other".
The notebook was corrected so that these rules null only the affected column, not the whole row. An earlier version had nulled entire rows for Age and CreditScore.

## 4. Duplicates
The raw file had 15 exact duplicate rows. They were dropped, leaving 10,000 rows.

## 5. Categorical cleaning
Geography and Gender were cleaned in the same order:
1. Trim whitespace and convert to lowercase.
2. Replace unknown tokens with missing.
3. Correct typos.
4. Fill remaining missing values with the mode.
Geography raw variants included France, Franc, " France", FRANCE, france, Spain, Spainn, Germany, Germny, USA, "???", Unknown and NaN. After cleaning only france (5,025), germany (2,501) and spain (2,474) remain. Gender raw variants included male, female, m, f, u, other and nan. "m" and "f" were mapped to male and female. After cleaning, only male and female remain.

## 6. Target cleaning
`Exited` had 5 missing values. Missing targets cannot be imputed sensibly, so those 5 rows were dropped and the target was cast to integer. The cleaned data has 9,995 rows.

## 7. Scope note on imputation
All imputation values above (medians, modes) were computed on the full cleaned dataset in Sections 4-7, before the train/validation/test split in Section 15. The leakage-safe steps that are fitted on training data only are described in leakage_safe_processing.md.
