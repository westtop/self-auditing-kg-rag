# Self-Auditing KG-RAG — reproduction package

Reproduction package for the paper *Auditing the Evidence, Not the Answer: Making
Literature Disagreement Visible in Knowledge-Graph RAG*.

## What this is

The pipeline detects disagreements between reported parameter values in a small
agricultural literature, audits each flagged pair, attaches a categorical
Confidence Level (Verified / Disputed / Conflicted) to the evidence, and shows
those levels to the generator at answer time. This repository reproduces every
table and figure reported in the paper.

## Layout

Everything lives in the repository root, so the notebook resolves its three data
folders to `.` with no configuration.

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
| `answers_regenerated.json` | generated answers, five conditions |
| `judge_scores_regenerated.json` | judge scores, three judges |
| `t3_cap_diagnostic_v2.json` | peer-list cap diagnostic (Section 5.6) |
| `tables_v2/` | reported tables (CSV) |
| `figures_v2/` | reported figures (PNG and PDF) |

## Running it

```bash
pip install -r requirements.txt
jupyter lab Reproduce_SelfAuditingGraphRAG_v2.ipynb
```

The reported run used Python 3.12.7; the exact package versions are recorded in
`repro_out/MANIFEST_v2.json` under `environment`, and `requirements.txt` is left
unpinned so that the notebook records whatever versions the reader actually used.

Run the cells in order. The notebook writes to `repro_out/` and reuses the API
caches that are already there, so **a full pass over the deposited caches makes no
API calls**. Regenerating the answers or the judge scores from scratch does; set
`OPENAI_API_KEY` (and `ANTHROPIC_API_KEY` for judge J2) in the environment if you
want to do that.

**Never hard-code an API key in a notebook cell.** The notebook reads keys from
the environment.

The local open-weight judge (J3, `meta-llama-3.1-8b-instruct`) is pinned to a
named checkpoint served at `http://localhost:1234/v1`. The notebook raises an
error rather than substituting a different model.

## A note on what is *not* here

Earlier runs of this pipeline left artefacts in `repro_out/` that the deposited
notebook neither reads nor writes — among them a `MANIFEST.json` from a previous
version. They are not included, because a stale manifest sitting beside the
current one invites the reader to quote numbers that are not the ones reported.
`MANIFEST_v2.json` is the manifest for the reported run, and it is the only one.

## What is not included

**The parsed source text of the 19 source documents is not redistributed.** Those
documents are copyrighted; only the extracted parameter triplets, which are our
own derived data, are included here. The notebook regenerates the parsed chunks
from the source PDFs when the reader has them. The 19 documents are listed in
Supplementary Table S1 of the paper.

## Licence

Code (`*.py`, `*.ipynb`) — MIT.
Derived data (`*.json`, `*.csv`) and figures — CC BY 4.0.

## Citation

Cite the paper. The archived release of this repository has its own DOI:

```
[Zenodo DOI]
```
