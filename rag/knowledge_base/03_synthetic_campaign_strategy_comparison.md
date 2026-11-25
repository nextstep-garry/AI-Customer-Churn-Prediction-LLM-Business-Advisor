---
doc_id: KB-03
title: Retention Strategy Comparison (Observed Data vs Not Available)
status: SAMPLE FRAMEWORK - NO CAMPAIGN OUTCOMES EXIST
synthetic: true
source: Observed figures from churn_raw.csv scored with XGBoost v1.0.0; everything about campaign results is explicitly marked Not available
created: 2026-10-06
version: 0.1
---

# Retention Strategy Comparison

> **SAMPLE - FOR DEMONSTRATION ONLY.** The project's dataset contains **no campaign outcomes**. This document therefore contains no invented results. It separates what the data can say (population sizes and observed churn) from what it cannot (cost, save rate, ROI, A/B results, customer lifetime value), and describes how the missing figures should be measured.

## 1. Observed from the dataset
Candidate strategies refer to the plays in KB-01. Population counts come from the 9,773 scored customers.

| Candidate strategy | Target group | Customers in group |
|---|---|---|
| Re-engagement (Play A) | High risk AND inactive | 1,242 |
| Re-engagement, wider (Play A) | Medium risk AND inactive | 680 |
| Product engagement review (Play B) | High risk AND 1 product | 1,294 |
| Combined priority review (Play C) | High risk AND inactive AND 1 product | 951 |
| Investigate first (Play D) | High risk AND 3 or 4 products | 294 |
| Manual review (Play E) | High risk, neither inactive nor 1 product | 129 |

Groups overlap: a customer can be in both Play A and Play B. The 951 customers in Play C are counted inside both the 1,242 and the 1,294.

Observed churn for the underlying behaviours, from historical data:

| Behaviour | Customers | Observed churn |
|---|---|---|
| Inactive | 4,730 | 26.9% |
| 1 product | 4,989 | 27.6% |
| Inactive AND 1 product | 2,470 | 36.6% |

Overall risk distribution: Low 6,746 (69.0%), Medium 1,313 (13.4%), High 1,714 (17.5%). Overall observed churn is 20.4%.

## 2. Not available in the dataset
| Item | Status |
|---|---|
| Campaign cost per customer | Not available |
| Save rate (customers retained by an action) | Not available |
| Return on investment | Not available |
| A/B test results | Not available |
| Customer lifetime value | Not available |
| Customer response or opt-out rates | Not available |

None of these may be estimated from the churn dataset. Observed churn in a group shows how many left, not how many an action would save.

## 3. How to measure the missing figures
1. Pick a target group (for example the 951 customers in Play C).
2. Randomly split it into a treated group and a **control group that receives no action**.
3. Run the action on the treated group only, for a fixed period.
4. Compare churn between the two groups.

Definitions:
- **Incremental retained customers** = (control churn rate - treated churn rate) x treated group size.
- **Cost per retained customer** = total campaign cost / incremental retained customers.
- **ROI** = (incremental retained customers x value per retained customer - total campaign cost) / total campaign cost.

The value per retained customer comes from the business (for example from lifetime value). It is not in this dataset.

## 4. Results template (to be filled with real data)
| Strategy | Group size | Control churn | Treated churn | Cost | ROI |
|---|---|---|---|---|---|
| Re-engagement | Not available | Not available | Not available | Not available | Not available |
| Product engagement review | Not available | Not available | Not available | Not available | Not available |
| Combined priority review | Not available | Not available | Not available | Not available | Not available |

## 5. How the assistant should answer "compare strategies"
1. State the group size for each strategy using section 1.
2. State that effectiveness, cost and ROI are not available and cannot be derived from the churn data.
3. Compare only on things the data supports: how many customers each strategy would reach, how much the groups overlap, and the observed churn of the underlying behaviour.
4. Suggest the control-group design in section 3 as the way to get real answers.
5. Never state a save rate, cost or ROI. If a user asks for a ranking, say it would be a judgement without outcome data.

## 6. Optional: a synthetic demonstration scenario
If a demonstration with numbers is wanted, create a separate file whose every figure is labelled SYNTHETIC and whose filename says so. It must never be stored in this document or presented as measured results.
