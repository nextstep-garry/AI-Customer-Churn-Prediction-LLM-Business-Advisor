---
doc_id: KB-04
title: Observed Data Findings and Model Facts
status: FACTUAL - DERIVED FROM THE PROJECT DATA AND MODEL FILES
synthetic: false
source: data/datasets/f8667fa0b55847c18a4bd432053c96d7 (summary.json, results.csv) and models/model_manifest.json
created: 2026-10-06
version: 0.1
---

# Observed Data Findings and Model Facts

## 1. The dataset
- File: churn_raw.csv, a bank-customer file with 10,015 uploaded rows.
- After cleaning: 15 duplicates removed, 227 rows rejected as invalid, **9,773 customers scored**. Outcomes (`Exited`) are known for 9,768 of them.
- Overall observed churn: **20.4%**.
- **Caveat:** this is the same file the model was trained on, so the model's performance on it is optimistic. Observed churn rates by segment are plain counts of who left and are not affected by this. For model accuracy use the held-out figures in section 4.

## 2. Observed churn by segment
| Segment | Customers | Observed churn |
|---|---|---|
| France | 4,906 | 16.2% |
| Germany | 2,453 | 32.5% |
| Spain | 2,414 | 16.6% |
| Female | 4,438 | 25.1% |
| Male | 5,335 | 16.5% |
| Age 18-29 | 1,605 | 7.4% |
| Age 30-39 | 4,252 | 11.0% |
| Age 40-49 | 2,560 | 30.7% |
| Age 50-59 | 844 | 56.2% |
| Age 60+ | 512 | 28.1% |
| 1 product | 4,989 | 27.6% |
| 2 products | 4,466 | 7.7% |
| 3 products | 259 | 82.2% |
| 4 products | 59 | 100% |
| Active member | 5,043 | 14.2% |
| Inactive member | 4,730 | 26.9% |
| Inactive AND 1 product | 2,470 | 36.6% |
| Germany AND inactive | 1,230 | 41.1% |

The 3-product and 4-product groups are small. Treat them with caution.

## 3. The model
- XGBoost classifier, version 1.0.0, trained on 13 features built from 10 raw customer fields.
- Default threshold for flagging a customer: raw score of 0.50. In the sample dataset 3,027 customers (31.0%) are flagged.
- Risk tiers on the raw score: Low below 0.50, Medium 0.50 to 0.60, High 0.60 and above. Counts: Low 6,746, Medium 1,313, High 1,714.
- On the held-out test set, observed churn was at most 13.6% for raw scores below 0.50, about 24.6% for 0.50 to 0.60, and 53% to 85% at 0.60 and above.
- A calibrated variant (v1.1.0) uses tiers of 15% and 40% on a real probability scale.

## 4. Held-out model performance
Test set of 1,999 customers, threshold 0.5:
| Metric | Value |
|---|---|
| ROC-AUC | 0.8605 |
| Recall | 0.749 |
| Precision | 0.510 |
| F1 | 0.607 |
| Accuracy | 0.802 |
| Validation ROC-AUC (used for model selection) | 0.8664 |

## 5. What drives the model
Average absolute SHAP value across the scored customers (higher means a bigger influence on scores):
| Driver | Importance |
|---|---|
| Number of products | 0.400 |
| Age | 0.392 |
| Active member | 0.191 |
| Country: Germany | 0.124 |
| Gender | 0.115 |
| Account balance | 0.076 |
| Zero balance flag | 0.035 |
| Balance / salary ratio | 0.035 |
| Credit score | 0.022 |
| Estimated salary | 0.019 |

Tenure, Spain and credit-card ownership have very small influence. SHAP values show what the model used, not what causes churn.

## 6. Known limitations (from the model manifest)
1. The raw XGBoost score is not a calibrated probability: mean predicted 0.424 against observed churn of 0.204 on the test set.
2. The 0.5 threshold is a grid-boundary artifact of the notebook's threshold sweep, not a tuned optimum.
3. Winsorisation clips discrete features such as NumOfProducts to 3.5, so 3-product and 4-product customers are treated alike.
