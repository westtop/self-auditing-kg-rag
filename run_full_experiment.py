#!/usr/bin/env python3
"""
Full Experiment Runner — Self-Auditing GraphRAG Pipeline
========================================================
논문: Reliable Knowledge Triplet Extraction from Sparse Agricultural Documents

이 스크립트는 전체 실험을 순서대로 실행합니다:
  1. GPT-4o 트리플렛 추출 (Step 1)
  2. Rule-based Conflict Detection (Step 2)
  3. LLM Context Auditor (Step 3) — GPT-4o 사용
  4. Knowledge Graph Construction (Step 4)
  5. GraphRAG Inference + Sample Queries (Step 5)
  6. Injection Test CDR 측정
  7. Gold Standard 대비 Precision/Recall/F1 계산
  8. 결과 요약 JSON + 논문용 표 출력

사용법:
  pip install openai pymupdf
  export OPENAI_API_KEY="sk-..."
  python run_full_experiment.py

PDF 파일들을 pdfs/ 폴더에 넣어주세요.
"""

import os
import sys
import json
import time
from collections import defaultdict

# Import pipeline modules
from pipeline import (
    SelfAuditingPipeline, ConflictDetector, ContextAuditor,
    KnowledgeGraph, GraphRAGInference, THRESHOLDS
)

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False


# ============================================================
# Step 3 Enhanced: GPT-4o Context Auditor
# ============================================================
class GPT4oContextAuditor:
    """Step 3: Uses GPT-4o to verify conflicts contextually."""

    AUDIT_PROMPT = """You are an expert in wasabi hydroponic cultivation.
A triplet was flagged as a potential conflict. Evaluate whether this is:
- TRUE CONFLICT (T1: numeric range conflict, T2: unit mismatch, T3: conditional contradiction)
- FALSE POSITIVE (the value is actually correct given the condition)

Triplet:
  Subject: {subject}
  Predicate: {predicate}
  Object: {object}
  Unit: {unit}
  Condition: {condition}
  Source: {paper}

Conflict detected: {conflict_type}
Details: {details}

Domain thresholds:
{thresholds}
{context_section}
Respond in JSON:
{{
  "verdict": "TRUE_CONFLICT" or "FALSE_POSITIVE",
  "final_type": "T1" or "T2" or "T3" or null,
  "confidence": 0.0 to 1.0,
  "reasoning": "explanation"
}}"""

    def __init__(self, client):
        self.client = client

    @staticmethod
    def _find_other_conditioned_values(subject, triplet_id, all_triplets, unit=None, max_items=8):
        """For T3 auditing specifically: find other triplets sharing the same
        subject that DO have a condition specified, so the auditor can
        actually check whether the flagged (condition-missing) value is
        consistent with what's reported for other conditions -- the genuine
        "conditional contradiction" check the paper describes for T3.

        Added 2026-07-13: previously `all_triplets` was accepted as a
        parameter by `audit()` but never referenced anywhere in the prompt,
        so the model had zero visibility into other triplets and could only
        fall back to a generic "is this within the domain threshold" check
        for T3 candidates -- which is what T1 checks, not what T3 means.
        That gap, not any change in this prompt over time, is why T3
        confirmations kept coming back near zero on real 770-corpus runs.

        `unit` (also added 2026-07-13): matching on subject name alone let
        through category errors like comparing air_temp reported in "Degree
        hour" (a cumulative heat-accumulation metric) against genuine air_temp
        values in "°C" -- the same kind of category error already found and
        fixed for T2, recurring here. When `unit` is given, peers whose unit
        is not the SAME physical quantity (per ConflictDetector.units_compatible)
        are excluded from the comparison list.
        """
        if not all_triplets:
            return []
        results = []
        for t in all_triplets:
            if t.get("id") == triplet_id:
                continue
            if t.get("subject") != subject:
                continue
            if not t.get("condition"):
                continue
            if unit and not ConflictDetector.units_compatible(unit, t.get("unit", "")):
                continue
            results.append({
                "object": t.get("object", ""),
                "unit": t.get("unit", ""),
                "condition": t.get("condition", ""),
                "paper": t.get("paper", "")
            })
            if len(results) >= max_items:
                break
        return results

    def audit(self, triplet, conflict_type, details, all_triplets):
        if conflict_type is None:
            return {
                "triplet_id": triplet["id"],
                "original_conflict": None,
                "confidence": 0.95,
                "audit_action": "pass",
                "audit_log": "No conflict. High confidence.",
                "final_conflict": None
            }

        subject = triplet.get("subject", "")

        # T3_candidate: three earlier prompt-wording variants of a full GPT-4o
        # verdict step, run against this exact same 28-candidate set on
        # 2026-07-13, produced 0, then 27, then 9 confirmed T3s with zero
        # change to the underlying facts -- and a later attempt to ask GPT
        # only "is this parameter type generally condition-sensitive" turned
        # out to be near-tautological with Step 2's own candidate rule
        # (passing Step 2 already implies a conditioned peer exists, so the
        # answer was almost always "yes"), landing on 28/28 by construction
        # rather than genuine judgment. Redesigned 2026-07-13 into two parts:
        #   1. Deterministic evidence-count gate: if the comparison peers for
        #      this parameter all trace back to a SINGLE source paper, there
        #      isn't enough independent corroboration to confidently call
        #      this confirmed OR cleared -> "Undefined" (see
        #      KnowledgeGraph.add_triplet's confidence_level mapping).
        #   2. Otherwise, GPT-4o judges a narrower, non-tautological question:
        #      is the SPECIFIC comparison actually a fair/relevant one (same
        #      kind of claim), not "is this parameter type sensitive in
        #      general" -- restoring genuine Step 3 verification power (it
        #      can still clear a false match) without the two failure modes
        #      above.
        if conflict_type == "T3_candidate":
            other_values = self._find_other_conditioned_values(
                subject, triplet.get("id"), all_triplets, unit=triplet.get("unit", "")
            )
            distinct_papers = {ov["paper"] for ov in other_values if ov.get("paper")}

            if len(distinct_papers) <= 1:
                explanation = self._explain_t3(triplet, other_values, undefined=True)
                return {
                    "triplet_id": triplet["id"],
                    "original_conflict": conflict_type,
                    "confidence": 0.5,
                    "audit_action": "undefined",
                    "audit_log": explanation,
                    "final_conflict": "T3_undefined"
                }

            return self._judge_t3_relevance(triplet, other_values)

        threshold_info = json.dumps(THRESHOLDS.get(subject, {}), ensure_ascii=False)

        prompt = self.AUDIT_PROMPT.format(
            subject=subject,
            predicate=triplet.get("predicate", ""),
            object=triplet.get("object", ""),
            unit=triplet.get("unit", ""),
            condition=triplet.get("condition", "none"),
            paper=triplet.get("paper", ""),
            conflict_type=conflict_type,
            details=json.dumps(details, ensure_ascii=False, default=str),
            thresholds=threshold_info,
            context_section=""
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            result = json.loads(response.choices[0].message.content)
            time.sleep(0.3)

            audit_action = "flag" if result["verdict"] == "TRUE_CONFLICT" else "clear"

            # Enforce the invariant that final_conflict is only ever populated
            # when the model actually confirmed a true conflict. Diagnostic run
            # (2026-07-13, 9 T3_candidate cases, back when T3 still went through
            # this same verdict path) showed GPT-4o sometimes fills "final_type"
            # with the ORIGINAL candidate type (echoing the input) even when
            # "verdict" is FALSE_POSITIVE and its own "reasoning" text argues no
            # contradiction exists -- the AUDIT_PROMPT schema never explicitly
            # said final_type must be null unless verdict is TRUE_CONFLICT, so
            # the model's behavior here isn't "wrong" per the prompt as written,
            # but any code that counts confirmed conflicts by final_conflict
            # alone (as checkpoint 7 does) will silently overcount false
            # positives as confirmed. Fixing here, at the source, keeps that
            # invariant true for every downstream consumer regardless of prompt
            # wording changes. (T3_candidate no longer reaches this code path,
            # but T1/T2 still do, so the guard stays.)
            final_conflict = result.get("final_type") if audit_action == "flag" else None

            return {
                "triplet_id": triplet["id"],
                "original_conflict": conflict_type,
                "confidence": result.get("confidence", 0.5),
                "audit_action": audit_action,
                "audit_log": result.get("reasoning", ""),
                "final_conflict": final_conflict
            }

        except Exception as e:
            return {
                "triplet_id": triplet["id"],
                "original_conflict": conflict_type,
                "confidence": 0.5,
                "audit_action": "error",
                "audit_log": f"GPT-4o audit error: {str(e)}",
                "final_conflict": conflict_type
            }

    def _explain_t3(self, triplet, other_values, undefined=False):
        """Generate a short, evidence-citing rationale for a T3_candidate whose
        outcome has already been decided (either "Undefined" due to a single-
        source evidence gate, or handled by _judge_t3_relevance for the
        genuine verdict case). This method never decides TRUE_CONFLICT vs
        FALSE_POSITIVE itself. Falls back to a templated (non-GPT)
        explanation if the API call fails, so a transient error never
        silently drops or corrupts the already-decided label."""
        subject = triplet.get("subject", "")
        obj = triplet.get("object", "")
        unit = triplet.get("unit", "")
        paper = triplet.get("paper", "")

        if not other_values:
            return (
                f"Undefined: '{subject}' = {obj} {unit} was reported in {paper} without a "
                f"condition, and no unit-compatible condition-bearing peer was found "
                f"elsewhere in the corpus, so there isn't enough evidence to confirm or "
                f"clear a conditional contradiction."
            )

        comparison_lines = "\n".join(
            f"  - {ov['object']} {ov['unit']} (condition: {ov['condition']}, source: {ov['paper']})"
            for ov in other_values
        )

        if undefined:
            prompt = (
                "You are an expert in wasabi hydroponic cultivation. A parameter was "
                "reported WITHOUT a condition. Other reports of the SAME parameter under "
                "DIFFERENT conditions exist, but ALL of them trace back to a single source "
                "paper -- so this has been labeled 'Undefined' (not enough independent "
                "corroboration to confirm or clear a conditional contradiction), not TRUE "
                "or FALSE. Do not argue for either verdict. Your only job is to write a "
                "short (2-3 sentence) explanation of why the evidence is insufficient here, "
                "citing the single source and its values.\n\n"
                f"Flagged: {subject} = {obj} {unit} (source: {paper})\n\n"
                f"Comparison values (single source):\n{comparison_lines}\n\n"
                "Respond with the explanation text only -- no JSON, no preamble, no verdict."
            )
        else:
            prompt = (
                "You are an expert in wasabi hydroponic cultivation. A parameter was reported "
                "WITHOUT a condition, and other reports of the SAME parameter under DIFFERENT "
                "specified conditions exist in this corpus (listed below). This has ALREADY "
                "been determined to be a genuine conditional contradiction (T3) -- do not "
                "re-evaluate or question whether it is one. Your only job is to write a short "
                "(2-3 sentence) explanation, citing at least one specific comparison value "
                "below, of why not knowing the condition for the flagged value could lead to "
                "a wrong conclusion.\n\n"
                f"Flagged: {subject} = {obj} {unit} (source: {paper})\n\n"
                f"Other reported values for the same parameter under different conditions:\n"
                f"{comparison_lines}\n\n"
                "Respond with the explanation text only -- no JSON, no preamble, no verdict."
            )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250
            )
            time.sleep(0.3)
            return response.choices[0].message.content.strip()
        except Exception as e:
            label = "Undefined (insufficient independent evidence)" if undefined else "T3 (confirmed)"
            return (
                f"{label}: '{subject}' = {obj} {unit} was reported in {paper} without a "
                f"condition, while other conditioned reports of '{subject}' in the corpus "
                f"include {other_values[0]['object']} {other_values[0]['unit']} "
                f"(condition: {other_values[0]['condition']}). "
                f"[GPT-4o explanation generation failed, using fallback text: {e}]"
            )

    def _judge_t3_relevance(self, triplet, other_values):
        """For a T3_candidate with 2+ independent source papers backing the
        comparison: ask GPT-4o to judge whether the SPECIFIC comparison
        values are actually a fair, relevant comparison for the flagged
        value -- NOT whether this general type of parameter is "usually"
        condition-sensitive (that broader question was found to be near-
        tautological with the Step 2 candidate rule itself, since passing
        Step 2 already implies a conditioned peer exists, and produced
        unstable near-100% confirmation regardless of case specifics).

        This restores a genuine verification role for Step 3: it can still
        clear a candidate where the comparison peers, despite matching on
        subject and unit, are not actually comparable in kind (e.g. a
        substrate-sterilization side-effect measurement being compared
        against a general optimal-range recommendation)."""
        subject = triplet.get("subject", "")
        comparison_lines = "\n".join(
            f"  - {ov['object']} {ov['unit']} (condition: {ov['condition']}, source: {ov['paper']})"
            for ov in other_values
        )
        prompt = (
            "You are an expert in wasabi hydroponic cultivation. A parameter was reported "
            "WITHOUT a condition. Other reports of the SAME parameter (same subject, same "
            "physical unit) under DIFFERENT specified conditions exist in this corpus, from "
            "at least two independent sources, and are listed below.\n\n"
            "Your job is NOT to judge whether this general TYPE of parameter is 'usually' "
            "condition-sensitive -- that question is too broad and was found to produce "
            "unreliable results (it is almost always true in the abstract). Instead, judge "
            "whether the SPECIFIC comparison values below are actually a FAIR, RELEVANT "
            "comparison for the flagged value: are they really describing the same KIND of "
            "claim (e.g., both a general cultivation recommendation, or both the same kind "
            "of specific experimental measurement), or is at least one of them a different "
            "kind of claim / different experimental purpose that only coincidentally shares "
            "the same subject and unit (e.g., a substrate-sterilization side-effect "
            "measurement being compared against a general optimal-range recommendation)?\n\n"
            "  - If the comparison values ARE a fair, relevant comparison AND they report "
            "meaningfully different values for the same kind of claim: verdict = "
            "TRUE_CONFLICT (the missing condition creates genuine ambiguity).\n"
            "  - If the comparison values are NOT actually a fair/relevant comparison "
            "(different kind of claim or purpose), or they are relevant but not meaningfully "
            "different in practice: verdict = FALSE_POSITIVE (no genuine contradiction).\n\n"
            f"Flagged: {subject} = {triplet.get('object')} {triplet.get('unit', '')} "
            f"(predicate: {triplet.get('predicate', '')}, source: {triplet.get('paper', '')})\n\n"
            f"Comparison values:\n{comparison_lines}\n\n"
            "Respond in JSON:\n"
            "{\n"
            '  "verdict": "TRUE_CONFLICT" or "FALSE_POSITIVE",\n'
            '  "confidence": 0.0 to 1.0,\n'
            '  "reasoning": "explanation citing which comparison value(s) you relied on, and whether they are (or are not) a fair comparison in kind"\n'
            "}"
        )
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400
            )
            result = json.loads(response.choices[0].message.content)
            time.sleep(0.3)
            is_conflict = result.get("verdict") == "TRUE_CONFLICT"
            return {
                "triplet_id": triplet["id"],
                "original_conflict": "T3_candidate",
                "confidence": result.get("confidence", 0.5),
                "audit_action": "flag" if is_conflict else "clear",
                "audit_log": result.get("reasoning", ""),
                "final_conflict": "T3" if is_conflict else None
            }
        except Exception as e:
            return {
                "triplet_id": triplet["id"],
                "original_conflict": "T3_candidate",
                "confidence": 0.5,
                "audit_action": "error",
                "audit_log": f"GPT-4o T3 relevance judgment error: {str(e)}",
                "final_conflict": "T3_candidate"
            }


# ============================================================
# Precision / Recall / F1 Calculation
# ============================================================
def compute_precision_recall_f1(gpt_triplets, gold_triplets, threshold=0.7):
    """Compare GPT-4o extracted triplets against Gold Standard."""
    def triplet_key(t):
        return (t.get("subject", ""), t.get("predicate", ""), str(t.get("object", "")))

    gold_keys = {triplet_key(t) for t in gold_triplets}
    gpt_keys = {triplet_key(t) for t in gpt_triplets}

    tp = len(gold_keys & gpt_keys)
    fp = len(gpt_keys - gold_keys)
    fn = len(gold_keys - gpt_keys)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "total_gpt": len(gpt_triplets),
        "total_gold": len(gold_triplets),
        "TP": tp, "FP": fp, "FN": fn,
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1": round(f1, 3),
        "FP_examples": [t for t in gpt_triplets if triplet_key(t) not in gold_keys][:5],
        "FN_examples": [t for t in gold_triplets if triplet_key(t) not in gpt_keys][:5]
    }


# ============================================================
# Main Experiment Runner
# ============================================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gs_path = os.path.join(base_dir, "triplets_gold_standard.json")
    inj_path = os.path.join(base_dir, "injection_test.json")

    print("=" * 70)
    print("  Self-Auditing GraphRAG — Full Experiment")
    print("  논문: Reliable Knowledge Triplet Extraction")
    print("=" * 70)

    # ---- Phase A: GPT-4o Extraction (if API available) ----
    gpt_triplets = None
    if HAS_OPENAI and os.environ.get("OPENAI_API_KEY"):
        print("\n[Phase A] GPT-4o Triplet Extraction (Step 1)")
        from step1_gpt4o_extraction import main as run_step1
        step1_result = run_step1()
        gpt_triplets = step1_result["triplets"]

        # Save GPT-4o triplets in pipeline format
        gpt_path = os.path.join(base_dir, "triplets_gpt4o_extracted.json")
        print(f"  GPT-4o 추출 완료: {len(gpt_triplets)}건 → {gpt_path}")
    else:
        print("\n[Phase A] GPT-4o 사용 불가 — Gold Standard 사용")
        print("  (OpenAI API 키가 없거나 openai 패키지 미설치)")

    # ---- Phase B: Load Gold Standard ----
    with open(gs_path, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    gold_triplets = gold_data["triplets"]
    print(f"\n[Phase B] Gold Standard: {len(gold_triplets)}건")

    # ---- Phase C: Precision/Recall/F1 (GPT-4o vs Gold) ----
    if gpt_triplets:
        print("\n[Phase C] GPT-4o vs Gold Standard — Precision/Recall/F1")
        before_metrics = compute_precision_recall_f1(gpt_triplets, gold_triplets)
        print(f"  검증 전 (Before Self-Auditing):")
        print(f"    추출 총 건수: {before_metrics['total_gpt']}")
        print(f"    TP={before_metrics['TP']}, FP={before_metrics['FP']}, FN={before_metrics['FN']}")
        print(f"    Precision={before_metrics['Precision']}, Recall={before_metrics['Recall']}, F1={before_metrics['F1']}")
    else:
        before_metrics = None

    # ---- Phase D: Full Pipeline (Step 2~5) ----
    print("\n[Phase D] Self-Auditing Pipeline (Steps 2-5)")
    pipeline = SelfAuditingPipeline()

    # Use GPT-4o triplets if available, else Gold Standard
    working_triplets = gpt_triplets if gpt_triplets else gold_triplets

    # Step 2
    print("\n  [Step 2] Rule-based Conflict Detection")
    step2 = pipeline.run_step2_detection(working_triplets)
    detected = [r for r in step2 if r["conflict_type"] is not None]
    print(f"    Detected: {len(detected)}/{len(working_triplets)}")

    # Step 3: Use GPT-4o auditor if available
    print("\n  [Step 3] Context Auditor")
    if HAS_OPENAI and os.environ.get("OPENAI_API_KEY"):
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        auditor = GPT4oContextAuditor(client)
        audits = []
        for t, s2 in zip(working_triplets, step2):
            audit = auditor.audit(t, s2["conflict_type"], s2.get("details") or {}, working_triplets)
            audits.append(audit)
        print(f"    GPT-4o Context Auditor 완료: {len(audits)}건")
    else:
        audits = pipeline.run_step3_audit(working_triplets, step2)
        print(f"    Heuristic Auditor 사용: {len(audits)}건")

    # Step 4
    print("\n  [Step 4] Knowledge Graph Construction")
    kg = KnowledgeGraph()
    for t, audit in zip(working_triplets, audits):
        kg.add_triplet(t, audit)
    stats = kg.get_stats()
    print(f"    Nodes: {stats['total_nodes']}, Edges: {stats['total_edges']}")
    print(f"    Uncertain: {stats['uncertain_edges']}, Avg Confidence: {stats['avg_confidence']:.3f}")

    # Step 5
    print("\n  [Step 5] GraphRAG Inference")
    inference = GraphRAGInference(kg)
    queries = ["DO", "water_temp", "EC", "AITC", "PPFD", "air_temp"]
    query_results = {}
    for q in queries:
        result = inference.answer_query(q)
        query_results[q] = result
        print(f"    {q}: {len(result.get('sources', []))} sources, conf={result.get('min_confidence', 'N/A')}")

    # ---- Phase E: Injection Test ----
    print("\n[Phase E] Injection Test CDR")
    with open(inj_path, 'r', encoding='utf-8') as f:
        inj_data = json.load(f)
    injections = inj_data["injections"]

    cdr_summary, cdr_details = pipeline.run_injection_test(injections)
    print(f"\n  {'Type':<10} {'TP':<5} {'FN':<5} {'Total':<7} {'CDR':<10}")
    print(f"  {'-'*37}")
    for ctype in ["T1", "T2", "T3", "overall"]:
        s = cdr_summary[ctype]
        print(f"  {ctype:<10} {s['TP']:<5} {s['FN']:<5} {s['total']:<7} {s['CDR']:<10}")

    # ---- Phase F: Gold Standard FPR ----
    print("\n[Phase F] Gold Standard FPR")
    fpr = pipeline.run_gold_standard_fpr(gold_triplets)
    print(f"  FPR: {fpr['FPR']} ({fpr['FP_count']}/{fpr['total']})")

    # ---- Phase G: After-audit metrics (if GPT-4o) ----
    if gpt_triplets:
        # Filter out low-confidence triplets
        audited_triplets = []
        for t, audit in zip(gpt_triplets, audits):
            if audit["confidence"] >= 0.5:
                audited_triplets.append(t)
        after_metrics = compute_precision_recall_f1(audited_triplets, gold_triplets)
        print(f"\n[Phase G] 검증 후 (After Self-Auditing):")
        print(f"  추출 건수: {after_metrics['total_gpt']} (from {before_metrics['total_gpt']})")
        print(f"  TP={after_metrics['TP']}, FP={after_metrics['FP']}, FN={after_metrics['FN']}")
        print(f"  Precision={after_metrics['Precision']}, Recall={after_metrics['Recall']}, F1={after_metrics['F1']}")
    else:
        after_metrics = None

    # ---- Phase H: Export Results ----
    print("\n" + "=" * 70)
    print("[결과 저장]")

    results = {
        "experiment_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_version": "v1.2",
        "data": {
            "gold_standard_count": len(gold_triplets),
            "gpt4o_extracted_count": len(gpt_triplets) if gpt_triplets else "N/A",
            "injection_count": len(injections)
        },
        "step2_injection_cdr": cdr_summary,
        "step2_gold_fpr": fpr,
        "step3_audit_summary": {
            action: sum(1 for a in audits if a["audit_action"] == action)
            for action in set(a["audit_action"] for a in audits)
        },
        "step4_kg_stats": stats,
        "before_audit_metrics": before_metrics,
        "after_audit_metrics": after_metrics,
    }

    result_path = os.path.join(base_dir, "experiment_results.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"  → {result_path}")

    # Neo4j export
    cypher_path = os.path.join(base_dir, "neo4j_import.cypher")
    with open(cypher_path, 'w', encoding='utf-8') as f:
        f.write(kg.export_for_neo4j())
    print(f"  → {cypher_path}")

    # ---- Print Paper Tables ----
    print("\n" + "=" * 70)
    print("논문용 표 (Paper Tables)")
    print("=" * 70)

    print("\n[표 1] Injection Test CDR — Rule-based Detector")
    print(f"{'충돌 유형':<15} {'TP':<5} {'FN':<5} {'탐지 대상':<8} {'CDR':<10}")
    print("-" * 45)
    for ctype in ["T1", "T2", "T3", "overall"]:
        s = cdr_summary[ctype]
        label = {"T1": "T1 수치 범위", "T2": "T2 단위 불일치", "T3": "T3 조건부 모순", "overall": "전체 CDR"}[ctype]
        print(f"{label:<15} {s['TP']:<5} {s['FN']:<5} {s['total']:<8} {s['CDR']:<10}")

    print(f"\n[표 2] Gold Standard FPR: {fpr['FPR']} ({fpr['FP_count']}/{fpr['total']})")

    if before_metrics and after_metrics:
        print(f"\n[표 3] 검증 전/후 트리플렛 추출 품질 비교")
        print(f"{'지표':<15} {'검증 전':<12} {'검증 후':<12} {'향상폭':<12}")
        print("-" * 50)
        for key in ["Precision", "Recall", "F1"]:
            b = before_metrics[key]
            a = after_metrics[key]
            print(f"{key:<15} {b:<12.3f} {a:<12.3f} {'+' if a-b>=0 else ''}{a-b:<12.3f}")

    print(f"\n[KG 통계] Nodes={stats['total_nodes']}, Edges={stats['total_edges']}, "
          f"Uncertain={stats['uncertain_edges']}, Avg Conf={stats['avg_confidence']:.3f}")

    print("\n실험 완료!")
    return results


if __name__ == "__main__":
    main()
