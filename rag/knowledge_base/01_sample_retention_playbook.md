---
doc_id: KB-01
title: Sample Retention Playbook
status: SAMPLE - FOR DEMONSTRATION ONLY
synthetic: true
source: Written for the Churn Intelligence Assistant demo, based on findings in KB-04 (churn_raw.csv + XGBoost v1.0.0)
created: 2026-10-06
version: 0.1
---

# Sample Retention Playbook

> **SAMPLE - FOR DEMONSTRATION ONLY.** This document is synthetic. It was written for the Churn Intelligence Assistant demonstration and does **not** represent any real company's retention playbook. The interventions below are *hypotheses* suggested by patterns in the data. The dataset contains no campaign results, so **it does not show that any of these actions work.**

## 1. Purpose and how to use it
This playbook maps a customer's risk tier and observable behaviour to a *suggested starting point* for a retention conversation. Every suggestion requires human review (see KB-02, policy SP-02) before anyone contacts a customer.

## 2. What the data shows (observed, historical)
Observed churn rates in the sample dataset (9,773 scored customers; overall churn 20.4%). Full tables are in KB-04.

| Group | Customers | Observed churn |
|---|---|---|
| Customers with 1 product | 4,989 | 27.6% |
| Customers with 2 products | 4,466 | 7.7% |
| Customers with 3 products | 259 | 82.2% (small sample) |
| Customers with 4 products | 59 | 100% (very small sample) |
| Inactive members | 4,730 | 26.9% |
| Active members | 5,043 | 14.2% |
| Inactive AND 1 product | 2,470 | 36.6% |
| Active AND 1 product | 2,519 | 18.8% |
| Inactive AND 2+ products | 2,260 | 16.4% |

These are correlations in historical data. They show who left, not why, and not what would have kept them.

## 3. Risk tiers used by this playbook
The default model (XGBoost v1.0.0) uses interim tiers on its **raw score**: Low below 0.50, Medium 0.50 to 0.60, High 0.60 and above. The raw score is a ranking score, **not a probability**. In the sample dataset: Low 6,746 customers, Medium 1,313, High 1,714.

## 4. Sample plays
Plays are chosen by risk tier plus **account behaviour** (activity status and number of products). Plays never use Gender, Geography or Age (see KB-02, policy SP-04).

### Play A - Re-engagement (sample)
- **When:** Medium or High risk AND inactive member.
- **Hypothesis:** An inactive member who is flagged may respond to a reason to use the account again.
- **Candidate actions (sample):** a check-in call or message; a reminder of unused account features; an invitation to a service review.
- **Size in sample data:** 1,242 High-risk inactive customers; 680 Medium-risk inactive customers.

### Play B - Product engagement review (sample)
- **When:** Medium or High risk AND exactly 1 product.
- **Hypothesis:** One-product customers churn more often than two-product customers (27.6% vs 7.7%), so a review of whether another product genuinely suits them may help. It must be a suitability conversation, not a push to sell.
- **Size in sample data:** 1,294 High-risk customers with 1 product.

### Play C - Priority review: inactive AND 1 product (sample)
- **When:** High risk AND inactive AND 1 product.
- **Why prioritised:** This overlap has the highest observed churn among the combinations examined (36.6%).
- **Candidate actions (sample):** combine Plays A and B in a single human-led conversation.
- **Size in sample data:** 951 customers.

### Play D - Investigate, do not apply standard offers (sample)
- **When:** High risk AND 3 or 4 products.
- **Why:** Observed churn is very high (82.2% and 100%) but the groups are small (259 and 59 customers), and the model treats 3 and 4 products alike because of how the training data was clipped. This pattern may reflect a product or data issue rather than a customer-level problem.
- **Action (sample):** route to product or analytics review before any customer outreach.
- **Size in sample data:** 294 High-risk customers (235 with 3 products, 59 with 4).

### Play E - Manual profile review (sample)
- **When:** High risk but NOT inactive and NOT 1 product.
- **Action (sample):** no default action; a reviewer examines the customer's profile and the model's top drivers first.
- **Size in sample data:** 129 customers.

### Play F - Monitor / no action (sample)
- **When:** Low risk, or Medium risk with none of the triggers above.
- **Action (sample):** routine service; re-score at the next scheduled run.

## 5. Findings that call for investigation, not customer targeting
These segment patterns are real in the data but must not be used to treat individual customers differently:
- Customers in Germany churn at 32.5%, about double France (16.2%) and Spain (16.6%).
- Customers aged 50-59 churn at 56.2%, versus 7.4% for ages 18-29.
- Women churn at 25.1% versus 16.5% for men.

A sensible use is a market-level or product-level review (for example, why one country performs differently), not different offers or service levels for individuals based on country, age or gender.

## 6. What this playbook cannot tell you
Not available in the dataset: which action works, expected save rate, cost of any action, or customer lifetime value. See KB-03 for how to measure these properly.

## 7. How the assistant should use this document
- Always call these plays "sample" and say they are not proven.
- Quote the observed numbers from this document or KB-04 only; never invent effectiveness figures.
- Always add that a human must review before any customer action.
