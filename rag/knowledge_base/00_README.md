---
doc_id: KB-00
title: Knowledge Base Index for the Churn Intelligence Assistant
created: 2026-10-06
version: 0.1
---

# Churn Intelligence Assistant: Knowledge Base

This folder holds the documents the assistant retrieves from. Each document states whether it is real project material, derived from the data, or sample content.

| Doc ID | File | Type | Supports question |
|---|---|---|---|
| KB-01 | 01_sample_retention_playbook.md | SAMPLE (synthetic) | 3 Retention action |
| KB-02 | 02_sample_policy_and_eligibility_guidelines.md | SAMPLE (synthetic) | 5 Policies and recommendations |
| KB-03 | 03_synthetic_campaign_strategy_comparison.md | SAMPLE framework, no invented results | 7 Compare strategies |
| KB-04 | 04_observed_data_findings.md | Factual, derived from the data and model files | 1, 2, 4, 6 (context) |
| PD | project_docs/ | Real project documentation | 8 Questions about the model and system |

## Contents of project_docs/
| File | Describes |
|---|---|
| README.md | How the app works, API, running it, security notes |
| CHANGES.md | What was implemented against the audit requirements, and what is still open |
| models_README.md | The model artifacts and how to replace them |
| model_card.json | Model version, metrics, ranges, tiers, limitations |
| model_manifest.json | Artifact hashes, features, threshold, library versions |
| calibration_report.json | Calibration results for the calibrated variant |

## Rules for ingestion and use
1. Keep each document's front-matter block with its chunks, so the status (sample or factual) travels with the text.
2. Split on headings; each section is written to make sense on its own.
3. When the assistant answers from a sample document it must say it is sample content and not a real company policy (KB-02, SP-11).
4. Questions 1, 2, 4 and 6 are answered mainly by calling the backend tools (customer lookup, segment breakdown, what-if scoring). These documents only add context.
5. There is no real company behind this project. If one is added, replace KB-01 to KB-03 with the company's own documents and keep KB-04 and project_docs/.
