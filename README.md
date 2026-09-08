# Self-Auditing KG-RAG — reproduction package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22654812.svg)](https://doi.org/10.5281/zenodo.22654812)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](#licence)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Reproduction package for the paper *Auditing the Evidence, Not the Answer: Making
Literature Disagreement Visible in Knowledge-Graph RAG*.

## What this is

The pipeline detects disagreements between reported parameter values in a small
agricultural literature, audits each flagged pair, attaches a categorical
Confidence Level (Verified / Disputed / Conflicted) to the evidence, and shows
those levels to the generator at answer time. This repository reproduces every
table and figure reported in the paper.

## Layout

Everything required for reproduction lives between the repository root and
`repro_out/`. The notebook resolves its three data folders to `.` and its cache
directory to `repro_out/`, with no configuration — **moving these files breaks
that resolution**, so clone the repository rather than downloading files
individually.

### Repository root

| File | Role |
|---|---|
| `Reproduce_SelfAuditingGraphRAG_v2.ipynb` | the reproduction notebook — run this |
| `pipeline.py` | rule-stage `ConflictDetector` (T1/T2/T3), the threshold and unit tables, and the `KnowledgeGraph` container |
| `run_full_experiment.py` | the LLM context auditor (`GPT4oContextAuditor`) |
| `step1_gpt4o_extraction.py` | triplet extraction, used only when re-extracting from the PDFs |
| `pipeline_hf.py`, `metrics.py` | audit and aggregation for the strawberry corpus (Section 4.7) |
| `strawberry_config.py` | domain configuration for that corpus |
| `gpt4o_extracted_merged.json` | 770 extracted parameter triplets (19 source documents) |
| `triplets_gold_standard.json` | 60-item curated Gold Standard |
| `causal_relations.json` | 374 curated causal relations |
| `judge_inputs.json` | the 20 evaluation queries and the generated answers |
| `RQ2_judge_scores_canonical.json` | the originally recorded judge scores, kept for the before/after comparison in Part 9b |
| `extractions.json` | the independent 76-paper strawberry corpus (Section 4.7) |
| `requirements.txt` | dependencies (unpinned — see *Running it*) |
| `CITATION.cff` | citation metadata |

### `repro_out/`

Audit caches, judge scores, tables and figures for the reported run.

> **Which files are the reported ones.** The reported run regenerated the answers
> and the judge scores; `MANIFEST_v2.json` records this as
> `reporting_source: regenerated` and `judging_source: regenerated`, and the
> notebook selects them with `REPORT = "regenerated"` in Part 9b. The two files
> without the `_regenerated` suffix are the **earlier recorded run**, kept only so
> that Part 9b can show the before/after comparison. Do not quote numbers from
> them.

| File | Contents |
|---|---|
| `MANIFEST_v2.json` | run manifest: input hashes, package versions, and every reported figure. **The authoritative record of the reported run.** |
| `audited_triplets.json` | the 770 triplets with their Confidence Levels (Verified 734, Conflicted 36) |
| `step2_results.json` | rule-stage detection output |
| `step3_audits.json` | LLM audit verdicts for the flagged pairs |
| `gold_audits.json` | audits of the 60 Gold Standard items (false-alarm analysis, Section 4.1) |
| `injection_audits_corpus.json` | injection benchmark, ordinary-corpus reference (Section 4.2) |
| `injection_audits_oracle.json` | injection benchmark, oracle reference — the upper bound |
| **`answers_regenerated.json`** | **the reported answers** for the five conditions |
| `answers.json` | answers from the earlier recorded run — kept for the Part 9b comparison, **not the reported answers** |
| `answers_b5.json` | the B5 stability re-runs |
| **`judge_scores_regenerated.json`** | **the reported judge scores** behind Tables 3 and 5 |
| `judge_scores.json` | judge scores from that earlier run — kept for the same comparison, **not the reported scores** |
| `judge_scores_std.json` | evidence-aware rubric scores (FTH / ANR / CTX), Table 6 |
| `ur_scores.json` | judged uncertainty-reporting scores over the 14 disputed queries, Table 4 |
| `scores_b5.json` | judge scores for the B5 stability re-runs |
| `stability_runs.json` | the stability run index |
| `closed_book.json` | closed-book control (Part 11) |
| `case_c1_illustration.json` | the worked case discussed in Section 5.3 |
| `t3_cap_diagnostic_v2.json` | peer-list cap diagnostic (Section 5.6) |
| `tables_v2/` | 15 result tables (CSV) |
| `figures_v2/` | 13 figures (PNG and PDF) |

### Which deposited file corresponds to which table or figure in the paper

The deposited tables and figures are numbered by the analysis that produced them,
not by their order in the paper. The paper's numbered exhibits map as follows.

| In the paper | Deposited file |
|---|---|
| Table 1 (the five generation conditions) | descriptive — no data file |
| Table 2 (injection benchmark) | `tables_v2/T6_injection.csv` |
| Table 3 (conventional answer quality) | `tables_v2/T11_judge_by_baseline.csv` |
| Table 4 (judged uncertainty reporting) | computed in Part 9d from `ur_scores.json` |
| Table 5 (confirmatory family) | `tables_v2/T13_significance.csv` |
| Table 6 (evidence-aware rubric) | computed in Part 9f from `judge_scores_std.json` |
| Figure 1 (injection by conflict type) | `figures_v2/F4_injection.png` / `.pdf` |
| Figure 2 (answer quality by judge) | `figures_v2/F8_judge_by_baseline.png` / `.pdf` |
| Figure 3 (disclosure and uncertainty reporting) | `figures_v2/F13_disclosure_and_UR.png` / `.pdf` |
| Figure 4 (retrieved subgraph) | `figures_v2/F7_retrieved_subgraph.png` / `.pdf` |

The remaining files in `tables_v2/` and `figures_v2/` are supporting analyses
quoted in the text: corpus composition (`T1`), the threshold and unit reference
tables (`T2`, Supplementary S2), rule-stage detection (`T3`, `F2`), the Confidence
Level distribution (`T4`, `F3`), confirmed conflict-type combinations (`T5`), the
Gold Standard false-alarm analysis (`T7`), recall against the similarity threshold
(`T8`, `F5`), the knowledge graph (`T9`, `F6`), baseline retrieval statistics
(`T10`), scores by criterion (`T12`, `F9`), the pipeline funnel (`F1`), and the
strawberry corpus (`T14`, `T15`, `F10`–`F12`).

## Running it

```bash
git clone https://github.com/westtop/self-auditing-kg-rag.git
cd self-auditing-kg-rag
pip install -r requirements.txt
jupyter lab Reproduce_SelfAuditingGraphRAG_v2.ipynb
```

Run the cells in order. The notebook writes to `repro_out/` and reuses the API
caches that are already there, so **a full pass over the deposited caches makes no
API calls**. Regenerating the answers or the judge scores from scratch does; set
`OPENAI_API_KEY` (and `ANTHROPIC_API_KEY` for judge J2) in the environment if you
want to do that.

> **Never hard-code an API key in a notebook cell.** The notebook reads keys from
> the environment.

The reported run used Python 3.12.7 on Windows 11; the exact package versions are
recorded in `repro_out/MANIFEST_v2.json` under `environment`. `requirements.txt`
is deliberately left unpinned, so that a reproduction run records whatever
versions the reader actually used rather than silently inheriting ours.

The local open-weight judge (J3, `meta-llama-3.1-8b-instruct`) is pinned to a
named checkpoint served at `http://localhost:1234/v1`. The notebook raises an
error rather than substituting a different model.

## Notes on the code

`pipeline.py` also contains earlier entry points — `SelfAuditingPipeline`,
`ContextAuditor`, `GraphRAGInference` and `add_relational_edge` — retained for the
standalone script. **The reported experiments do not call them.** Confidence
Levels for the reported run are assigned by `assign_confidence()` in the notebook
(Part 4), which defines the three levels of Section 3.5 only — Verified, Disputed,
Conflicted — and maps an undetermined T3 gate result to Verified. The
`"Undefined"` branch in `add_relational_edge` is unreachable in this work, and no
item in `audited_triplets.json` carries it.

The LLM context audit is not in `pipeline.py`, which does not import `openai`; it
is `GPT4oContextAuditor` in `run_full_experiment.py`.

## A note on what is *not* here

Earlier runs of this pipeline left artefacts in `repro_out/` that the deposited
notebook neither reads nor writes — among them a `MANIFEST.json` from a previous
version. They are not included, because a stale manifest sitting beside the
current one invites the reader to quote numbers that are not the ones reported.
`MANIFEST_v2.json` is the manifest for the reported run, and it is the only one.

**The parsed source text of the 19 source documents is not redistributed.** Those
documents are copyrighted; only the extracted parameter triplets, which are our
own derived data, are included here. The notebook regenerates the parsed chunks
from the source PDFs when the reader has them. The 19 documents are listed in
Supplementary Table S1 of the paper and in the paper's reference list.

## Licence

Code (`*.py`, `*.ipynb`) — MIT.
Derived data (`*.json`, `*.csv`) and figures — CC BY 4.0.

## Citation

Cite the paper. The archived release of this repository has its own DOI:

```
https://doi.org/10.5281/zenodo.22654812
```

`CITATION.cff` in the repository root carries the machine-readable metadata.
