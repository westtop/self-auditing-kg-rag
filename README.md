# Self-Auditing KG-RAG

Reproducibility materials for **“Auditing the Evidence, Not the Answer: Evidence-Level Conflict Detection Makes Literature Disagreement Visible in Knowledge-Graph RAG.”**

## Overview

This repository contains the code, extracted evidence, audit outputs, generated answers, evaluation scores, tables, and figures used in the study.

The study evaluates **Self-Auditing KG-RAG**, a knowledge-graph RAG pipeline that audits conflicts among evidence items before generation and exposes the resulting Confidence Levels to the generator.

## Repository contents

- `Reproduce_SelfAuditingGraphRAG_v2.ipynb` — main reproduction notebook.
- `pipeline.py` — conflict detection, auditing, knowledge-graph, and inference code.
- `gpt4o_extracted_merged.json` — extracted parameter triplets from the study corpus.
- `triplets_gold_standard.json` — curated Gold Standard.
- `causal_relations.json` — curated causal relations used for graph retrieval.
- `judge_inputs.json` — inputs used for LLM-based evaluation.
- `RQ2_judge_scores_canonical.json` — canonical judge scores.
- `extractions.json` — extracted data used in the supplementary strawberry stability analysis.
- `answers.json`, `answers_b5.json` — generated answers.
- `audited_triplets.json`, `gold_audits.json` — audit outputs.
- `injection_audits_corpus.json`, `injection_audits_oracle.json` — injection-benchmark audit outputs.
- `judge_scores.json`, `scores_b5.json`, `ur_scores.json` — evaluation scores.
- `step2_results.json`, `step3_audits.json` — detector and audit-stage outputs.
- `stability_runs.json` — strawberry stability analysis outputs.
- `MANIFEST_v2.json` — reproducibility manifest.

The `figures_v2/` directory contains the figures generated for the paper, and `tables_v2/` contains the corresponding result tables.

## Reproduction

The notebook is organized around the frozen artefacts used in the reported experiments. It can regenerate the reported tables and figures and includes verification steps for:

1. retrieval identity across the controlled B3/B4/B5 comparison;
2. the judge-free disclosure criterion and its per-query audit trail; and
3. the joint Holm adjustment used for the confirmatory family.

Some pipeline stages can make external API calls when rerun from scratch. The notebook uses the stored artefacts for the reported results unless a rerun is explicitly requested.

## Source PDFs

The source PDFs are not redistributed in this repository. Their bibliographic information is provided in Supplementary Table S1 of the paper.

The extracted triplets used for the reported experiments are provided in `gpt4o_extracted_merged.json`.

## Evaluation conditions

The main controlled comparison uses:

- **B3:** multi-hop KG-RAG without Confidence Levels shown;
- **B4:** the same retrieval with an explicit disagreement-reporting instruction;
- **B5:** the same retrieval with audited Confidence Levels shown.

All three conditions use the same generator and the same retrieval subgraph.

## Environment

The experiments use the package versions recorded in `MANIFEST_v2.json`.

API credentials are not included in this repository. API keys must be supplied through the appropriate environment variables when rerunning API-dependent stages.

## Citation

Please cite the associated paper when using these materials.

A DOI for the archived release will be added after the repository is registered with Zenodo.
