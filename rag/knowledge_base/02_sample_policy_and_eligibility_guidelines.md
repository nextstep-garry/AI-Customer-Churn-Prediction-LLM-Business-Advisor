---
doc_id: KB-02
title: Sample Policy and Eligibility Guidelines
status: SAMPLE - FOR DEMONSTRATION ONLY
synthetic: true
source: Written for the Churn Intelligence Assistant demo; model-governance points are based on the project's own documented limitations (project_docs/CHANGES.md, models/README.md, model_manifest.json)
created: 2026-10-06
version: 0.1
---

# Sample Policy and Eligibility Guidelines

> **SAMPLE POLICY - FOR DEMONSTRATION ONLY.** This document is synthetic and created for the Churn Intelligence Assistant demonstration. It does not represent an actual company's policy. Bracketed values such as [N] are placeholders that a real organisation would set.

## SP-01 Model output is decision support, not a decision
A churn score or risk tier must never automatically trigger a customer action, offer or contact. The model ranks customers by risk; people decide what to do.

## SP-02 Human review before any offer or outreach
A named reviewer must confirm the recommendation, the customer's eligibility (SP-03) and the proposed action before any offer or contact is made. The reviewer should look at the model's top drivers for that customer, not only the tier.

## SP-03 Sample eligibility rules
A customer may be considered for a retention action only if all of the following hold:
1. Risk tier is Medium or High (default model: raw score of 0.50 or more).
2. The customer has consented to the relevant type of contact.
3. The customer has not been contacted about retention within the last [N] days.
4. The customer is not already enrolled in another retention action.
5. The account is in good standing: [definition set by the business].
6. The proposed action's value is within the approved limit of [limit set by the business].

## SP-04 Sensitive attributes must not drive treatment
Gender, Geography and Age are inputs to the model, and the model uses them. They must **not** be used to decide eligibility, offer value, or service level for an individual customer. Segment reports by gender, country and age are for monitoring and fairness review only. A periodic review should compare flag rates and action rates across these groups and report differences for investigation.

## SP-05 A raw score is not a probability
The default model (XGBoost v1.0.0) outputs a raw ranking score. On the held-out test set it averaged 0.424 against an observed churn rate of 0.204, so a score of 0.55 does **not** mean a 55% chance of leaving. Describe it as a "risk score" or "risk tier". Only the calibrated model variant (v1.1.0, which fits a Platt calibrator) may be described in probability terms. When quoting a score, always state the model version and score type.

## SP-06 Drivers are not causes
SHAP drivers show which inputs moved the model's score for a customer. They do not prove why the customer will leave, and they must not be presented to the customer as reasons. Say "the model associates this with higher risk", not "this will cause churn".

## SP-07 Use held-out figures for performance claims
The sample dataset (churn_raw.csv) is the same file the model was trained on, so performance measured on it is optimistic. Quote the held-out test figures instead: ROC-AUC 0.8605, recall 0.749, precision 0.51 at the 0.5 threshold.

## SP-08 Small-sample rule
Do not build standard rules on groups that are too small to be reliable (set the minimum at [N] customers). In the sample data, the 3-product (259) and 4-product (59) groups are examples to treat with caution.

## SP-09 Data handling
Uploaded customer data is stored unencrypted under `data/` until deleted, so delete datasets when finished. Set `API_KEY` before exposing the API. The API has no user accounts; put it behind a proxy with TLS if used beyond a local demo.

## SP-10 Record keeping and learning
Log each recommendation, the reviewer's decision and the outcome. The prediction history already records scores; outcomes must be recorded separately. Without a control group this still cannot prove an action worked (see KB-03).

## SP-11 Rules for the assistant itself
1. Cite the document ID (for example KB-02, SP-04) for any policy it quotes.
2. Say that sample documents are sample. Never present them as real company policy.
3. If the knowledge base does not contain the answer, say so and name where it could be found. Never invent campaign results, costs or policies.
4. Never recommend an automatic customer action.
5. Describe raw scores as risk scores, not probabilities.

## Which situations these policies apply to
| Situation | Policies that apply |
|---|---|
| Recommending a retention action for one customer | SP-01, SP-02, SP-03, SP-04 |
| Explaining a prediction to a colleague or customer | SP-05, SP-06 |
| Reporting model accuracy | SP-07 |
| Comparing segments or building rules | SP-04, SP-08 |
| Uploading or storing customer files | SP-09 |
