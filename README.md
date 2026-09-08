# Self-Auditing KG-RAG — reproduction package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22654812.svg)](https://doi.org/10.5281/zenodo.22654812)

Reproduction package for the paper *Auditing the Evidence, Not the Answer: Making Literature Disagreement Visible in Knowledge-Graph RAG*.

## What this is

The pipeline detects disagreements between reported parameter values in a small
agricultural literature, audits each flagged pair, attaches a categorical
Confidence Level (Verified / Disputed / Conflicted) to the evidence, and shows
those levels to the generator at answer time. This repository contains the code
and deposited artefacts used to reproduce the tables and figures reported in
the paper.

## Layout

Everything required for reproduction is organized between the repository root and
`repro_out/`. Code, configuration, input data, and the reproduction notebook remain
in the repository root, while cached outputs, generated answers, judge scores,
tables, and figures are stored under `repro_out/`.

| File | Role |
|---|---|
| `Reproduce_SelfAuditingGraphRAG_v2.ipynb` | the reproduction notebook — run this |
| `pipeline.py` | detector, domain configuration tables, graph classes |
| `run_full_experiment.py` | the context auditor (`GPT4oContextAuditor`) |
| `step1_gpt4o_extraction.py` | triplet extraction, used only when re-extracting from the PDFs |
| `pipeline_hf.py`, `metrics.py` | audit and aggregation for the strawberry corpus (Section 4.7) |
| `gpt4o_extracted_merged.json` | 770 extracted parameter triplets (19 source documents) |
| `triplets_gold_standard.json` | 60-item curated Gold Standard |
| `causal_relations.json` | 374 curated causal relations |
| `judge_inputs.json` | 20 evaluation queries and the generated answers |
| `RQ2_judge_scores_canonical.json` | the originally recorded judge scores, kept for the before/after comparison in Part 9b |
| `extractions.json`, `strawberry_config.py` | the independent 76-paper strawberry corpus and its configuration (Section 4.7) |
| `repro_out/` | manifest, audit caches, judge scores, tables and figures |

### `repro_out/`

| Path | Contents |
|---|---|
| `MANIFEST_v2.json` | run manifest: input hashes, package versions, every reported figure. **This is the authoritative record of the reported run.** |
| `audited_triplets.json` | the 770 triplets with their Confidence Levels |
| `answers.json` | generated answers for the reported conditions |
| `answers_b5.json` | B5 answers from the reported run |
| `answers_regenerated.json` | regenerated answer cache used by the reproduction workflow |
| `judge_scores.json` | reported judge scores |
| `judge_scores_regenerated.json` | regenerated judge-score cache used by the reproduction workflow |
| `judge_scores_std.json` | evidence-aware judging outputs |
| `scores_b5.json` | B5 scoring output |
| `ur_scores.json` | uncertainty-reporting scores |
| `gold_audits.json` | Gold Standard audit results |
| `step2_results.json`, `step3_audits.json` | intermediate detector and audit outputs |
| `injection_audits_corpus.json` | corpus-referenced injection audit results |
| `injection_audits_oracle.json` | oracle-referenced injection audit results |
| `stability_runs.json` | sparse-sampling stability results |
| `t3_cap_diagnostic_v2.json` | peer-list cap diagnostic (Section 5.6) |
| `case_c1_illustration.json` | case-study illustration data |
| `closed_book.json` | closed-book diagnostic output |
| `figures_v2/` | reported figures (PNG and PDF) |
| `tables_v2/` | reported tables (CSV) |

## Running it

```bash
pip install -r requirements.txt
jupyter lab Reproduce_SelfAuditingGraphRAG_v2.ipynb
