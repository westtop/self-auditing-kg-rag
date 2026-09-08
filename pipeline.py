#!/usr/bin/env python3
"""
Self-Auditing GraphRAG Pipeline for Hydroponic Wasabi Management
================================================================
Version: v1.2
Paper: Reliable Knowledge Triplet Extraction from Sparse Agricultural Documents

Modules:
  Step 1: Triplet Extraction (from JSON / or GPT-4o in production)
  Step 2: Rule-based Conflict Detector (T1/T2/T3)
  Step 3: LLM Context Auditor (Confidence Score)
  Step 4: Knowledge Graph Construction (dict-based, exportable to Neo4j)
  Step 5: GraphRAG Inference (graph traversal + answer generation)
"""

import json
import re
import os
from collections import defaultdict
from datetime import datetime

# ============================================================
# DOMAIN THRESHOLDS (v1.2 — pilot test reflected)
# ============================================================
THRESHOLDS = {
    "water_temp":          {"min": 5,    "max": 20,   "unit": "°C",        "sources": ["Wikipedia", "World of Wasabi 2017"]},
    "growth_medium_temp":  {"min": 5,    "max": 23,   "unit": "°C",        "sources": ["Oguni et al. 2005", "pilot T1 fix"]},
    "air_temp":            {"min": 5,    "max": 30,   "unit": "°C",        "sources": ["Kazaoka et al. 2023"]},
    "pH":                  {"min": 5.5,  "max": 7.5,  "unit": "",          "sources": ["Yamashita 2024", "Bugbee 2004"]},
    "DO":                  {"min": 6,    "max": 14,   "unit": "mg/L",      "sources": ["Hoang 2019", "Soffer & Burger 1988"]},
    "EC":                  {"min": 0.05, "max": 3.7,  "unit": "dS/m",      "sources": ["Hoang 2019", "Hoagland 1/10~1/2"]},
    "PPF":                 {"min": 50,   "max": 300,  "unit": "μmol/m²/s", "sources": ["Tanaka et al. 2008"]},
    "PPFD":                {"min": 50,   "max": 900,  "unit": "μmol/m²/s", "sources": ["Taylor et al. 2025"]},
    "photoperiod":         {"min": 8,    "max": 16,   "unit": "hours",     "sources": ["Tanaka et al. 2008"]},
    "VPD":                 {"min": 0.3,  "max": 2.5,  "unit": "kPa",       "sources": ["Taylor et al. 2025"]},
}

# Unit conversion knowledge base
UNIT_CONVERSIONS = {
    ("dS/m", "mS/cm"):     1.0,
    ("mS/cm", "dS/m"):     1.0,
    ("mg/L", "ppm"):        1.0,
    ("ppm", "mg/L"):        1.0,
    ("°C", "°F"):           lambda c: c * 9/5 + 32,
    ("°F", "°C"):           lambda f: (f - 32) * 5/9,
    ("mg/kg", "g/kg"):      0.001,
    ("g/kg", "mg/kg"):      1000,
    ("mmol/kg", "mol/kg"):  0.001,
    ("mol/kg", "mmol/kg"):  1000,
    ("Pa", "kPa"):          0.001,
    ("kPa", "Pa"):          1000,
    ("μS/cm", "dS/m"):      0.001,
    ("dS/m", "μS/cm"):      1000,
    ("lux", "μmol/m²/s"):   None,  # non-linear, flagged
    ("ft", "m"):             0.3048,
    ("m", "ft"):             3.28084,
    ("mg/g", "mg/kg"):       1000,
    ("mg/kg", "mg/g"):       0.001,
    ("μg/L", "mg/L"):        0.001,
    ("mg/L", "μg/L"):        1000,
    ("minutes", "hours"):    1/60,
    ("hours", "minutes"):    60,
    # --- Real-corpus unit gaps found when re-auditing the 770-triplet Step2 run ---
    ("S/m", "dS/m"):            10,
    ("dS/m", "S/m"):             0.1,
    ("mmol/m²/s", "μmol/m²/s"):  1000,
    ("μmol/m²/s", "mmol/m²/s"):  0.001,
    ("mol/s/m²", "μmol/m²/s"):   1000000,
    ("μmol/m²/s", "mol/s/m²"):   0.000001,
    ("hPa", "kPa"):               0.1,
    ("kPa", "hPa"):               10,
    # "INCOMPATIBLE" (not None!) marks pairs that are NOT the same physical
    # quantity at all -- None elsewhere in this table means "same quantity,
    # but no linear factor exists" (e.g. lux -> μmol/m²/s, still a genuine
    # unit mismatch worth flagging). These four are different in kind, not
    # just notation, so they must NOT be treated as a T2 unit mismatch.
    ("%", "μmol/m²/s"):           "INCOMPATIBLE",  # PPF/PPFD reported as % of a device max
    ("% of saturation", "mg/L"):  "INCOMPATIBLE",  # DO reported as % saturation, not concentration
    ("months", "hours"):          "INCOMPATIBLE",  # photoperiod reported as a duration, not daily hours
    ("Degree hour", "°C"):        "INCOMPATIBLE",  # heat-accumulation metric, not instantaneous temperature
}

# Pure notation/formatting variants that mean the exact same physical unit as
# a KNOWN_UNIT_ALIASES/THRESHOLDS canonical unit (no value conversion needed).
# Found by auditing the raw unit strings in the 770-triplet GPT-4o extraction
# corpus against the canonical spellings used above (e.g. differing hyphen vs.
# slash notation, "µ" MICRO SIGN vs "μ" GREEK MU, "sec" vs "s", etc.).
UNIT_NOTATION_MAP = {
    "mg L−1": "mg/L",
    "mg L-1": "mg/L",
    "mg-liter-1": "mg/L",
    "dS m-1": "dS/m",
    "dS m−1": "dS/m",
    "dSm-1": "dS/m",
    "ms/cm": "mS/cm",
    "µS/cm": "μS/cm",
    "µmol m-2 s-1": "μmol/m²/s",
    "µmol-s-1-m-2": "μmol/m²/s",
    "µmol m-2s-1": "μmol/m²/s",
    "μmol m-2 s-1": "μmol/m²/s",
    "μmol m−2 s−1": "μmol/m²/s",
    "umol m-2 sec-1": "μmol/m²/s",
    "umol m^2 sec^-1": "μmol/m²/s",
    "mmol·m²·s⁻¹": "mmol/m²/s",
    "h": "hours",
    "h d-1": "hours",
    "h d−1": "hours",
    "hr": "hours",
}


def normalize_unit_notation(unit):
    """Map a raw extracted unit string to its canonical spelling, if it is a
    known pure-notation variant. Leaves genuinely different units untouched
    (those are handled by UNIT_CONVERSIONS instead)."""
    if not unit:
        return unit
    return UNIT_NOTATION_MAP.get(unit, unit)


KNOWN_UNIT_ALIASES = {
    "water_temp": ["°C", "°F"],
    "growth_medium_temp": ["°C", "°F"],
    "air_temp": ["°C", "°F"],
    "DO": ["mg/L", "ppm"],
    "EC": ["dS/m", "mS/cm", "μS/cm"],
    "AITC": ["mg/kg", "g/kg", "mg/g", "mg/g FW", "ratio"],
    "glucosinolates": ["mmol/kg DW", "mol/kg DW"],
    "PPF": ["μmol/m²/s", "lux"],
    "PPFD": ["μmol/m²/s", "lux"],
    "VPD": ["kPa", "Pa"],
    "altitude": ["m", "ft"],
    "photoperiod": ["hours", "minutes"],
}

# Units that are equivalent (1:1) and should NOT trigger T2
EQUIVALENT_UNITS = {
    frozenset({"mg/L", "ppm"}),
    frozenset({"dS/m", "mS/cm"}),
}


# ============================================================
# Step 2: Rule-based Conflict Detector
# ============================================================
class ConflictDetector:
    """Detects T1 (numeric range), T2 (unit mismatch), T3 candidates (condition-dependent)."""

    def __init__(self, thresholds=None):
        self.thresholds = thresholds or THRESHOLDS
        self.log = []

    @staticmethod
    def units_compatible(unit_a, unit_b):
        """True if unit_a and unit_b measure the SAME physical quantity (same
        unit, a known equivalent, or a pair with a defined conversion factor)
        -- False only if they are a known-INCOMPATIBLE pair (different kind
        of measurement entirely, e.g. "Degree hour" vs "°C"). If either unit
        is missing, or the pair is simply unknown (no entry either way), this
        returns True -- we only actively deny compatibility where we have
        explicit evidence, matching the same conservative discipline used by
        the T1/T2 checks. Added 2026-07-13: without this, T3_candidate
        generation and its comparison-value lookup only matched on subject
        name, so a triplet like air_temp reported in "Degree hour" (a
        cumulative heat-accumulation metric, not a temperature) could be
        compared against genuine °C air_temp values as if they were the same
        kind of fact -- the same category-error bug already found and fixed
        for T2, recurring in a different code path.
        """
        if not unit_a or not unit_b:
            return True
        a = normalize_unit_notation(unit_a)
        b = normalize_unit_notation(unit_b)
        if a == b:
            return True
        if frozenset({a, b}) in EQUIVALENT_UNITS:
            return True
        conv = UNIT_CONVERSIONS.get((a, b), UNIT_CONVERSIONS.get((b, a)))
        if conv == "INCOMPATIBLE":
            return False
        return True

    def detect(self, triplet, all_triplets=None):
        """Returns (conflict_type, details) or (None, None)."""
        subject = triplet.get("subject", "")
        obj_raw = triplet.get("object", "")
        unit = normalize_unit_notation(triplet.get("unit", ""))
        condition = triplet.get("condition")

        # --- T2: Unit mismatch detection ---
        t2_result = self._check_unit_mismatch(triplet, all_triplets)
        if t2_result:
            return ("T2", t2_result)

        # --- T1: Numeric range conflict (fixed domain threshold) ---
        t1_result = self._check_numeric_range(subject, obj_raw, unit)
        if t1_result:
            return ("T1", t1_result)

        # --- T1 (additional): direct cross-triplet range conflict. The check
        # above only catches a value that violates the fixed domain threshold
        # (e.g. outside the broad literature-wide min/max). It does NOT catch
        # two specific triplets that directly contradict each other while
        # each individually still falls inside that broad range. This check
        # compares the flagged triplet's reported value/range against every
        # OTHER unit-compatible triplet for the same subject that ALSO has NO
        # condition specified (both unconditioned -- if either side has a
        # condition, a difference is expected/explainable and belongs to T3,
        # not here). If the two ranges do not overlap AT ALL, that is a
        # genuine numeric conflict between two specific sources. Partial or
        # full overlap is NOT flagged -- different sources reporting
        # overlapping/compatible ranges is normal, not contradictory.
        t1_overlap_result = self._check_range_overlap_conflict(triplet, subject, obj_raw, unit, condition, all_triplets)
        if t1_overlap_result:
            return ("T1", t1_overlap_result)

        # --- T3 candidate: has numeric value, in thresholds, missing/vague condition,
        # AND at least one other triplet for the SAME subject DOES specify a condition.
        # This last requirement is deterministic and decided entirely here (Step 2 /
        # rule-based), not by the LLM auditor in Step 3. Without it, a triplet could be
        # labeled a "conditional contradiction candidate" even when no condition-bearing
        # data exists anywhere in the corpus to judge it against -- i.e. there would be
        # nothing for Step 3 to actually audit. Added 2026-07-13 so the candidate-vs-
        # confirmation split matches what the paper describes: Step 2 decides WHY
        # something is a candidate (deterministically), Step 3 decides WHETHER the
        # missing condition is material (semantic judgment only).
        if subject in self.thresholds and condition is None:
            nums = self._extract_numbers(obj_raw)
            if nums and all_triplets:
                has_conditioned_peer = any(
                    t.get("subject") == subject
                    and t.get("id") != triplet.get("id")
                    and t.get("condition")
                    and self.units_compatible(unit, normalize_unit_notation(t.get("unit", "")))
                    for t in all_triplets
                )
                if has_conditioned_peer:
                    return ("T3_candidate", {
                        "reason": "Numeric value for known parameter but no condition specified, "
                                  "and condition-specific values for this same parameter exist "
                                  "elsewhere in the corpus",
                        "subject": subject,
                        "value": obj_raw,
                        "note": "Requires LLM Context Auditor (Step 3) to judge materiality"
                    })

        return (None, None)

    def _check_numeric_range(self, subject, obj_raw, unit):
        if subject not in self.thresholds:
            return None
        threshold = self.thresholds[subject]
        nums = self._extract_numbers(str(obj_raw))
        if not nums:
            return None

        # Normalize unit if needed
        normalized_nums = self._normalize_to_threshold_unit(nums, unit, threshold["unit"], subject)
        if normalized_nums is None:
            # Reported in a unit that is not the same physical quantity as the
            # threshold (e.g. DO as "% of saturation" vs threshold's "mg/L") --
            # there is no valid way to compare against a min/max range, so this
            # is not a determinable T1 conflict either way. Skip rather than
            # comparing incomparable numbers.
            return None

        for val in normalized_nums:
            if val < threshold["min"] or val > threshold["max"]:
                return {
                    "reason": f"Value {val} outside range [{threshold['min']}, {threshold['max']}] {threshold['unit']}",
                    "subject": subject,
                    "value": val,
                    "threshold": threshold,
                    "original_unit": unit
                }
        return None

    def _check_range_overlap_conflict(self, triplet, subject, obj_raw, unit, condition, all_triplets):
        """Additional T1 check: direct disagreement between two specific
        UNCONDITIONED triplets for the same (unit-compatible) subject, even
        when both individually pass the fixed domain-threshold check. Only
        compares against peers that ALSO have no condition specified -- if
        either side has a condition, the difference is potentially explained
        by that condition and is T3's territory, not a T1 numeric conflict.
        Flags only when the two reported ranges have ZERO overlap; any
        partial or full overlap is treated as compatible, not conflicting."""
        if condition or not all_triplets:
            return None
        nums = self._extract_numbers(str(obj_raw))
        if not nums:
            return None
        my_min, my_max = min(nums), max(nums)

        for other in all_triplets:
            if other.get("id") == triplet.get("id"):
                continue
            if other.get("subject") != subject:
                continue
            if other.get("condition"):
                continue  # only compare like-for-like (both unconditioned)
            other_unit = normalize_unit_notation(other.get("unit", ""))
            if not self.units_compatible(unit, other_unit):
                continue
            other_nums_raw = self._extract_numbers(str(other.get("object", "")))
            if not other_nums_raw:
                continue
            if other_unit and other_unit != unit:
                other_nums = self._normalize_to_threshold_unit(other_nums_raw, other_unit, unit, subject)
                if other_nums is None:
                    continue
            else:
                other_nums = other_nums_raw
            other_min, other_max = min(other_nums), max(other_nums)

            if my_max < other_min or my_min > other_max:
                return {
                    "reason": f"Range [{my_min}, {my_max}] {unit} does not overlap at all with "
                              f"{other['id']}'s range [{other_min}, {other_max}] {unit} for the "
                              f"same unconditioned parameter",
                    "subject": subject,
                    "value": obj_raw,
                    "other_id": other["id"],
                    "this_range": [my_min, my_max],
                    "other_range": [other_min, other_max],
                    "unit": unit
                }
        return None

    def _check_unit_mismatch(self, triplet, all_triplets):
        subject = triplet.get("subject", "")
        unit = normalize_unit_notation(triplet.get("unit", ""))
        if not unit or subject not in KNOWN_UNIT_ALIASES:
            return None

        # Skip if unit pair is known equivalent (e.g. mg/L ≡ ppm, dS/m ≡ mS/cm)
        def _are_equivalent(u1, u2):
            return frozenset({u1, u2}) in EQUIVALENT_UNITS

        if subject in self.thresholds:
            primary_unit = self.thresholds[subject]["unit"]
            if unit and unit != primary_unit:
                if _are_equivalent(unit, primary_unit):
                    return None  # equivalent units, no conflict
                conversion_key = (unit, primary_unit)
                if conversion_key in UNIT_CONVERSIONS:
                    conv = UNIT_CONVERSIONS[conversion_key]
                    if conv == "INCOMPATIBLE":
                        return None  # different kind of measurement, not a mismatch
                    elif conv is None:
                        return {
                            "reason": f"Non-linear unit conversion: {unit} → {primary_unit}",
                            "subject": subject, "from_unit": unit, "to_unit": primary_unit
                        }
                    elif conv == 1.0:
                        return None  # 1:1 conversion, not a real mismatch
                    else:
                        return {
                            "reason": f"Unit mismatch: {unit} used instead of {primary_unit}",
                            "subject": subject, "from_unit": unit, "to_unit": primary_unit,
                            "conversion_factor": conv if not callable(conv) else "function"
                        }

        # Cross-check with other triplets: only flag when the two units are BOTH
        # known measures of the SAME physical quantity -- i.e. a defined
        # conversion relationship exists between them in UNIT_CONVERSIONS (in
        # either direction). Diagnostic run (2026-07-13, notebook checkpoint 4
        # after finally passing all_triplets for the first time this session)
        # showed that without this restriction, this check massively
        # overcounts T2 (44 -> 221) by flagging pairs like "mg/L vs % of
        # saturation" (DO), "°C vs Degree hour" (air_temp), "hours vs months"
        # (photoperiod), and "μmol/m²/s vs %" (PPFD) as unit mismatches --
        # these are not notation variants of the same fact, they are
        # different kinds of measurement (or, in the "hours vs months" case,
        # possibly a Step 1 extraction/labeling error) that happen to share a
        # subject label. Requiring a known conversion factor keeps this check
        # to its actual purpose: catching genuine same-quantity, different-
        # notation cases (e.g. dS/m vs μS/cm), matching the same discipline
        # already used in the domain-threshold check above.
        if all_triplets:
            for other in all_triplets:
                if other.get("id") == triplet.get("id"):
                    continue
                if other.get("subject") == subject and other.get("unit") and unit:
                    other_unit = normalize_unit_notation(other.get("unit"))
                    if other_unit != unit and not _are_equivalent(unit, other_unit):
                        conv_key = (unit, other_unit)
                        rev_key = (other_unit, unit)
                        if conv_key not in UNIT_CONVERSIONS and rev_key not in UNIT_CONVERSIONS:
                            # No known relationship between these units -- likely a
                            # different kind of measurement, not a notation mismatch.
                            continue
                        conv = UNIT_CONVERSIONS.get(conv_key, UNIT_CONVERSIONS.get(rev_key))
                        if conv == "INCOMPATIBLE":
                            continue  # different kind of measurement, not a mismatch
                        if conv == 1.0:
                            continue  # 1:1 conversion, not a real mismatch
                        return {
                            "reason": f"Unit differs from {other['id']}: {unit} vs {other_unit}",
                            "subject": subject, "from_unit": unit, "to_unit": other_unit,
                            "other_id": other["id"]
                        }
        return None

    def _normalize_to_threshold_unit(self, nums, source_unit, target_unit, subject):
        if not source_unit or source_unit == target_unit:
            return nums
        key = (source_unit, target_unit)
        if key in UNIT_CONVERSIONS:
            conv = UNIT_CONVERSIONS[key]
            if conv == "INCOMPATIBLE":
                return None  # not the same physical quantity -- cannot compare at all
            if conv is None:
                return nums
            if callable(conv):
                return [conv(n) for n in nums]
            return [n * conv for n in nums]
        return nums

    @staticmethod
    def _extract_numbers(text):
        text = str(text)
        # Handle ranges like "5~20", "0.6~3.7", "13.5~28.1"
        # A hyphen directly between two digits (no space) is a range separator
        # ("13.5-28.1" = 13.5 to 28.1), not a minus sign. A hyphen at the start
        # of a number or preceded by whitespace/non-digit IS a genuine minus
        # sign ("-3 C"). Neutralize only the range-separator case.
        text = re.sub(r'(?<=[0-9])-(?=[0-9])', ' ', text)
        # Strip "±X" error-margin suffixes so the margin isn't extracted as a
        # separate reported value (e.g., "11.4±2.4" -> keep 11.4, drop 2.4)
        text = re.sub(r'±\s*\d+\.?\d*', '', text)
        nums = re.findall(r'-?\d+\.?\d*', text)
        return [float(n) for n in nums] if nums else []


# ============================================================
# Step 3: LLM Context Auditor (simplified — uses rules + heuristics)
# ============================================================
class ContextAuditor:
    """Assigns Confidence Score and validates conflicts contextually.
    In production, this uses GPT-4o. Here we use heuristic rules."""

    def audit(self, triplet, conflict_type, conflict_details, all_triplets):
        result = {
            "triplet_id": triplet["id"],
            "original_conflict": conflict_type,
            "confidence": 0.95,
            "audit_action": "pass",
            "audit_log": "",
            "final_conflict": None
        }

        if conflict_type is None:
            # C = P0 * D * A = 0.95 * 1.0 * 1.0 = 0.95
            result["confidence"] = 0.95
            result["audit_log"] = "No conflict detected. High confidence."
            return result

        if conflict_type == "T1":
            # T1 confirmed — low confidence
            condition = triplet.get("condition")
            if condition:
                # Has condition → might be T3 (conditional contradiction)
                # C = P0 * D * A = 0.95 * 0.50 * 1.50 = 0.71
                result["final_conflict"] = "T3"
                result["confidence"] = 0.71
                result["audit_action"] = "reclassify_T1_to_T3"
                result["audit_log"] = (
                    f"T1 flag but condition present: '{condition}'. "
                    f"Reclassified to T3 (conditional contradiction). "
                    f"Details: {conflict_details.get('reason', '')}"
                )
            else:
                result["final_conflict"] = "T1"
                # C = P0 * D * A = 0.95 * 0.50 * 0.70 = 0.33
                result["confidence"] = 0.33
                result["audit_action"] = "flag_T1"
                result["audit_log"] = (
                    f"T1 CONFIRMED: {conflict_details.get('reason', '')}. "
                    f"Value significantly outside domain threshold."
                )

        elif conflict_type == "T2":
            # Check if values are equivalent after conversion
            result["final_conflict"] = "T2"
            # Default: T2 incompatible. C = P0 * D * A = 0.95 * 0.70 * 0.70 = 0.47
            result["confidence"] = 0.47
            from_unit = conflict_details.get("from_unit", "")
            to_unit = conflict_details.get("to_unit", "")
            conv = UNIT_CONVERSIONS.get((from_unit, to_unit))

            if conv is not None and conv != "function" and conv != "INCOMPATIBLE":
                nums = ConflictDetector._extract_numbers(str(triplet.get("object", "")))
                if nums:
                    converted = [n * conv if not callable(conv) else conv(n) for n in nums]
                    result["audit_log"] = (
                        f"T2: Unit mismatch {from_unit} → {to_unit}. "
                        f"Original: {nums}, Converted: {converted}. "
                        f"Values are equivalent after conversion → label mismatch only."
                    )
                    # C = P0 * D * A = 0.95 * 0.70 * 1.30 = 0.86
                    result["confidence"] = 0.86
                    result["audit_action"] = "flag_T2_convertible"
            else:
                result["audit_log"] = f"T2: Incompatible units {from_unit} vs {to_unit}."
                result["audit_action"] = "flag_T2_incompatible"

        elif conflict_type == "T3_candidate":
            # C = P0 * D * A = 0.95 * 0.70 * 1.00 = 0.67
            result["final_conflict"] = "T3"
            result["confidence"] = 0.67
            result["audit_action"] = "flag_T3_candidate"
            result["audit_log"] = (
                f"T3 candidate: {conflict_details.get('reason', '')}. "
                f"No condition specified for parameter {triplet.get('subject')}. "
                f"Expert review recommended."
            )

        return result


# ============================================================
# Step 4: Knowledge Graph (dict-based, NetworkX-compatible)
# ============================================================
class KnowledgeGraph:
    """Simple knowledge graph using adjacency list."""

    def __init__(self):
        self.nodes = {}       # {node_id: {type, label, ...}}
        self.edges = []       # [{source, target, predicate, confidence, source_id, ...}]
        self.adjacency = defaultdict(list)  # {node_id: [(target, edge_idx), ...]}

    def add_triplet(self, triplet, audit_result):
        subj = triplet["subject"]
        obj_str = f"{triplet['object']}"
        pred = triplet["predicate"]
        confidence = audit_result["confidence"]
        source_id = triplet.get("source_id", "unknown")

        # Add nodes
        if subj not in self.nodes:
            self.nodes[subj] = {"type": "parameter", "label": subj}

        obj_node_id = f"{subj}_{pred}_{source_id}"
        self.nodes[obj_node_id] = {
            "type": "value",
            "label": obj_str,
            "unit": triplet.get("unit", ""),
            "condition": triplet.get("condition", ""),
            "paper": triplet.get("paper", "")
        }

        # Confidence Level per paper Section 3.2 (categorical, not a raw
        # confidence threshold): PASS or FALSE_ALARM -> Verified; T3
        # confirmed -> Disputed; T1/T2 confirmed -> Conflicted. Based on
        # final_conflict alone so it works the same whether audit_result
        # comes from the heuristic ContextAuditor or GPT4oContextAuditor.
        final_conflict = audit_result.get("final_conflict")
        if final_conflict == "T3":
            confidence_level = "Disputed"
        elif final_conflict == "T3_undefined":
            # Added 2026-07-13: a T3_candidate whose comparison evidence
            # (condition-bearing peers for the same parameter) all traces
            # back to a SINGLE source paper. There isn't enough independent
            # corroboration to confidently call this either a confirmed
            # contradiction (Disputed) or a cleared false positive
            # (Verified), so it gets its own honest "insufficient evidence"
            # bucket instead of being forced into a binary.
            confidence_level = "Undefined"
        elif final_conflict in ("T1", "T2"):
            confidence_level = "Conflicted"
        else:
            confidence_level = "Verified"

        edge = {
            "source": subj,
            "target": obj_node_id,
            "predicate": pred,
            "confidence": confidence,
            "source_id": source_id,
            "triplet_id": triplet["id"],
            "conflict": audit_result.get("final_conflict"),
            "confidence_level": confidence_level,
            "uncertainty": confidence < 0.7,
            "audit_log": audit_result.get("audit_log", ""),
            # "relational" = parameter<->value star-edge from a single Step1
            # triplet (this method). "causal" = entity<->entity edge from
            # add_causal_edge() below. Kept distinct so callers/GraphRAG can
            # tell "this parameter has a recorded value" apart from "this
            # parameter/process causally affects that other one" — the paper's
            # Fig.1 gray-vs-green edge distinction.
            "edge_type": "relational"
        }
        edge_idx = len(self.edges)
        self.edges.append(edge)
        self.adjacency[subj].append((obj_node_id, edge_idx))
        self.adjacency[obj_node_id].append((subj, edge_idx))

    # Causal edges get a single flat, documented placeholder confidence —
    # NOT a GPT-4o self-reported score. Earlier drafts asked the extraction
    # model to self-rate its own confidence, but that conflated an unaudited,
    # poorly-calibrated LLM self-assessment with the paper's Self-Auditing
    # Confidence Level (which is only ever assigned by the actual Step 2/3
    # detect+audit pipeline). Fixed after code review, 2026-07-13.
    CAUSAL_EDGE_DEFAULT_CONFIDENCE = 0.75

    def add_causal_edge(self, relation):
        """
        Add a causal-relationship edge between two entity/process nodes
        (e.g. DO_low -> root_respiration_inhibition -> ATP_deficit), as
        opposed to add_triplet()'s parameter<->value star-edges.

        This is what the paper's Section 3.1 (Fig. 1 green edges) and
        Section 4.3 B4 example (seven-step causal chain) describe, and what
        was previously entirely missing from the code: add_triplet() alone
        can never connect two DIFFERENT parameters/entities to each other,
        so genuine multi-hop traversal (chain_length > 2) was structurally
        impossible. `relation` is expected to have the shape produced by
        step_causal_extraction.py:
            {
              "id": "CR-001",
              "cause": "DO_low", "cause_label": "Low dissolved oxygen",
              "relation": "inhibits",
              "effect": "root_respiration", "effect_label": "Root respiration",
              "evidence": "<supporting text span from source PDF>",
              "condition": "...or null",
              "source_id": "S04", "paper": "Soffer & Burger 1988, J ASHS"
            }
        confidence is ALWAYS CAUSAL_EDGE_DEFAULT_CONFIDENCE (not derived from
        Step2/Step3 — there is no numeric-range or unit check for a causal
        claim, and the extraction step no longer asks GPT-4o to self-rate a
        confidence, since that would conflate an unaudited LLM
        self-assessment with the paper's actual Self-Auditing Confidence
        Level). If a caller explicitly passes a "confidence" key in
        `relation` it is still honored (e.g. for a future Step-3-style
        causal auditor), but step_causal_extraction.py's current output does
        not include one. "Verified" here means "extracted, unaudited" — not
        "passed the T1/T2/T3 audit."
        """
        cause = relation["cause"]
        effect = relation["effect"]
        pred = relation.get("relation", "causes")
        confidence = relation.get("confidence", self.CAUSAL_EDGE_DEFAULT_CONFIDENCE)
        source_id = relation.get("source_id", "unknown")

        if cause not in self.nodes:
            self.nodes[cause] = {"type": "entity", "label": relation.get("cause_label", cause)}
        if effect not in self.nodes:
            self.nodes[effect] = {"type": "entity", "label": relation.get("effect_label", effect)}

        edge = {
            "source": cause,
            "target": effect,
            "predicate": pred,
            "confidence": confidence,
            "source_id": source_id,
            "triplet_id": relation.get("id", f"CR-{len(self.edges) + 1}"),
            "conflict": None,
            "confidence_level": "Verified",
            "uncertainty": confidence < 0.7,
            "audit_log": relation.get("evidence", ""),
            "edge_type": "causal"
        }
        edge_idx = len(self.edges)
        self.edges.append(edge)
        self.adjacency[cause].append((effect, edge_idx))
        self.adjacency[effect].append((cause, edge_idx))

    def get_neighbors(self, node_id, min_confidence=0.0, edge_types=None):
        """edge_types: optional iterable, e.g. ["causal"] or ["relational"].
        None (default) = no filter, matches all edge types."""
        results = []
        for target, edge_idx in self.adjacency.get(node_id, []):
            edge = self.edges[edge_idx]
            if edge["confidence"] < min_confidence:
                continue
            if edge_types is not None and edge.get("edge_type", "relational") not in edge_types:
                continue
            results.append((target, edge))
        return results

    def traverse_causal_chain(self, start_param, max_depth=5, min_confidence=0.3, edge_types=None):
        """BFS traversal from a parameter to find causal chains.
        edge_types: None = traverse both relational and causal edges
        (default, matches pre-fix behavior); ["causal"] = restrict to
        genuine causal-graph hops only, for queries that specifically want
        the paper's multi-hop causal-chain answers."""
        visited = set()
        queue = [(start_param, [start_param], [])]
        chains = []

        while queue:
            current, path, edge_path = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            neighbors = self.get_neighbors(current, min_confidence, edge_types=edge_types)
            for target, edge in neighbors:
                if target not in visited and len(path) < max_depth:
                    new_path = path + [target]
                    new_edge_path = edge_path + [edge]
                    chains.append({"path": new_path, "edges": new_edge_path})
                    queue.append((target, new_path, new_edge_path))

        return chains

    def get_stats(self):
        total_edges = len(self.edges)
        uncertain = sum(1 for e in self.edges if e["uncertainty"])
        conflicts = defaultdict(int)
        confidence_levels = defaultdict(int)
        edge_types = defaultdict(int)
        for e in self.edges:
            if e["conflict"]:
                conflicts[e["conflict"]] += 1
            confidence_levels[e.get("confidence_level", "Verified")] += 1
            edge_types[e.get("edge_type", "relational")] += 1
        return {
            "total_nodes": len(self.nodes),
            "total_edges": total_edges,
            "uncertain_edges": uncertain,
            "conflict_breakdown": dict(conflicts),
            "confidence_level_breakdown": dict(confidence_levels),
            "edge_type_breakdown": dict(edge_types),
            "avg_confidence": sum(e["confidence"] for e in self.edges) / max(total_edges, 1)
        }

    def export_for_neo4j(self):
        """Export Cypher statements for Neo4j import."""
        cypher = []
        for nid, props in self.nodes.items():
            safe_id = nid.replace(" ", "_").replace("/", "_")
            label = props.get("label", nid)
            ntype = props.get("type", "Entity")
            cypher.append(f'CREATE (n:{ntype} {{id: "{safe_id}", label: "{label}"}})')
        for edge in self.edges:
            src = edge["source"].replace(" ", "_").replace("/", "_")
            tgt = edge["target"].replace(" ", "_").replace("/", "_")
            pred = edge["predicate"]
            conf = edge["confidence"]
            cypher.append(
                f'MATCH (a {{id: "{src}"}}), (b {{id: "{tgt}"}}) '
                f'CREATE (a)-[:{pred.upper()} {{confidence: {conf}, source: "{edge["source_id"]}"}}]->(b)'
            )
        return "\n".join(cypher)


# ============================================================
# Step 5: GraphRAG Inference
# ============================================================
class GraphRAGInference:
    """Answers queries by traversing the knowledge graph."""

    def __init__(self, kg):
        self.kg = kg

    def answer_query(self, query_param, query_type="factual", edge_types=None):
        """Generate answer by graph traversal.
        edge_types=None (default): traverse both relational (parameter<->value)
        and causal (entity<->entity) edges — general factual queries.
        edge_types=["causal"]: restrict to the causal subgraph only, for a
        query that specifically wants a multi-hop causal-chain answer (the
        paper's B4-style "why does X happen" question)."""
        chains = self.kg.traverse_causal_chain(query_param, max_depth=5, min_confidence=0.3, edge_types=edge_types)

        if not chains:
            return {
                "answer": f"No information found for '{query_param}' in knowledge graph.",
                "sources": [],
                "confidence": 0.0,
                "chain_length": 0
            }

        # Collect evidence
        evidence = []
        sources = set()
        min_conf = 1.0
        warnings = []

        for chain in chains:
            for edge in chain["edges"]:
                node_info = self.kg.nodes.get(edge["target"], {})
                evidence.append({
                    "parameter": edge["source"],
                    "predicate": edge["predicate"],
                    "value": node_info.get("label", ""),
                    "unit": node_info.get("unit", ""),
                    "condition": node_info.get("condition", ""),
                    "paper": node_info.get("paper", ""),
                    "confidence": edge["confidence"],
                    "edge_type": edge.get("edge_type", "relational")
                })
                sources.add(edge["source_id"])
                min_conf = min(min_conf, edge["confidence"])
                if edge["uncertainty"]:
                    warnings.append(
                        f"⚠ Low confidence ({edge['confidence']:.2f}) for "
                        f"{edge['source']}→{edge['predicate']}: {edge['audit_log'][:80]}"
                    )

        # Build answer — causal edges read as "X predicate Y" (process chain),
        # relational edges read as "parameter predicate: value unit" (data point)
        answer_parts = []
        for ev in evidence:
            if ev["edge_type"] == "causal":
                part = f"  - {ev['parameter']} --[{ev['predicate']}]--> (causal) (신뢰도: {ev['confidence']:.2f})"
            else:
                part = f"  - {ev['parameter']} {ev['predicate']}: {ev['value']} {ev['unit']}"
                if ev['condition']:
                    part += f" (조건: {ev['condition']})"
                part += f" [{ev['paper']}] (신뢰도: {ev['confidence']:.2f})"
            answer_parts.append(part)

        answer_text = f"[{query_param}] 관련 지식 그래프 탐색 결과:\n" + "\n".join(answer_parts)

        if warnings:
            answer_text += "\n\n⚠ 불확실성 경고:\n" + "\n".join(warnings)

        return {
            "answer": answer_text,
            "evidence": evidence,
            "sources": list(sources),
            "min_confidence": min_conf,
            "chain_length": max(len(c["path"]) for c in chains) if chains else 0,
            "warnings": warnings
        }


# ============================================================
# Main Pipeline Orchestrator
# ============================================================
class SelfAuditingPipeline:
    """End-to-end pipeline: Extract → Detect → Audit → Build KG → Infer."""

    def __init__(self):
        self.detector = ConflictDetector()
        self.auditor = ContextAuditor()
        self.kg = KnowledgeGraph()
        self.inference = GraphRAGInference(self.kg)
        self.results = {
            "step2_detections": [],
            "step3_audits": [],
            "injection_results": [],
        }

    def load_triplets(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("triplets", [])

    def load_injections(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("injections", [])

    def run_step2_detection(self, triplets):
        """Step 2: Rule-based Conflict Detection."""
        results = []
        for t in triplets:
            conflict_type, details = self.detector.detect(t, triplets)
            results.append({
                "triplet_id": t["id"],
                "subject": t["subject"],
                "object": t["object"],
                "unit": t.get("unit", ""),
                "conflict_type": conflict_type,
                "details": details
            })
        self.results["step2_detections"] = results
        return results

    def run_step3_audit(self, triplets, step2_results):
        """Step 3: LLM Context Auditor."""
        audits = []
        for t, s2 in zip(triplets, step2_results):
            audit = self.auditor.audit(t, s2["conflict_type"], s2.get("details") or {}, triplets)
            audits.append(audit)
        self.results["step3_audits"] = audits
        return audits

    def run_step4_build_kg(self, triplets, audits):
        """Step 4: Build Knowledge Graph."""
        for t, audit in zip(triplets, audits):
            self.kg.add_triplet(t, audit)
        return self.kg.get_stats()

    def run_injection_test(self, injections):
        """Run Injection Test and compute CDR."""
        results = {"T1": {"TP": 0, "FN": 0}, "T2": {"TP": 0, "FN": 0}, "T3": {"TP": 0, "FN": 0}}
        details = []

        for inj in injections:
            conflict_type, conflict_details = self.detector.detect(inj)
            expected = inj["type"]
            detected = conflict_type is not None

            if detected:
                results[expected]["TP"] += 1
            else:
                results[expected]["FN"] += 1

            details.append({
                "id": inj["id"],
                "expected": expected,
                "detected_type": conflict_type,
                "detected": detected,
                "details": conflict_details
            })

        # Compute CDR
        summary = {}
        total_tp = 0
        total_fn = 0
        for ctype in ["T1", "T2", "T3"]:
            tp = results[ctype]["TP"]
            fn = results[ctype]["FN"]
            total = tp + fn
            cdr = tp / total if total > 0 else 0
            summary[ctype] = {"TP": tp, "FN": fn, "total": total, "CDR": f"{cdr*100:.1f}%"}
            total_tp += tp
            total_fn += fn

        overall_cdr = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        summary["overall"] = {"TP": total_tp, "FN": total_fn, "total": total_tp + total_fn, "CDR": f"{overall_cdr*100:.1f}%"}

        self.results["injection_results"] = {"summary": summary, "details": details}
        return summary, details

    def run_gold_standard_fpr(self, gold_triplets):
        """Compute FPR on Gold Standard (false positives)."""
        fp_list = []
        for t in gold_triplets:
            conflict_type, details = self.detector.detect(t, gold_triplets)
            if conflict_type is not None:
                fp_list.append({
                    "triplet_id": t["id"],
                    "subject": t["subject"],
                    "object": t["object"],
                    "detected_conflict": conflict_type,
                    "details": details
                })
        fpr = len(fp_list) / len(gold_triplets) if gold_triplets else 0
        return {"FP_count": len(fp_list), "total": len(gold_triplets), "FPR": f"{fpr*100:.1f}%", "FP_details": fp_list}

    def run_full_pipeline(self, gs_path, inj_path):
        """Run complete end-to-end pipeline."""
        print("=" * 70)
        print("Self-Auditing GraphRAG Pipeline v1.2")
        print("=" * 70)

        # Load data
        gold_triplets = self.load_triplets(gs_path)
        injections = self.load_injections(inj_path)
        print(f"\n[DATA] Gold Standard: {len(gold_triplets)} triplets")
        print(f"[DATA] Injection Test: {len(injections)} items")

        # Step 2: Detection on Gold Standard
        print("\n" + "-" * 50)
        print("[Step 2] Rule-based Conflict Detection (Gold Standard)")
        step2 = self.run_step2_detection(gold_triplets)
        detected_gs = [r for r in step2 if r["conflict_type"] is not None]
        print(f"  Detected conflicts: {len(detected_gs)}/{len(gold_triplets)}")

        # Gold Standard FPR
        fpr_result = self.run_gold_standard_fpr(gold_triplets)
        print(f"\n[FPR] Gold Standard FPR: {fpr_result['FPR']} ({fpr_result['FP_count']}/{fpr_result['total']})")
        if fpr_result['FP_details']:
            for fp in fpr_result['FP_details']:
                print(f"  FP: {fp['triplet_id']} ({fp['subject']}: {fp['object']}) → {fp['detected_conflict']}")

        # Step 2: Injection Test CDR
        print("\n" + "-" * 50)
        print("[Step 2] Injection Test CDR")
        cdr_summary, cdr_details = self.run_injection_test(injections)
        print(f"\n  {'Type':<8} {'TP':<5} {'FN':<5} {'Total':<7} {'CDR':<10}")
        print(f"  {'-'*35}")
        for ctype in ["T1", "T2", "T3", "overall"]:
            s = cdr_summary[ctype]
            print(f"  {ctype:<8} {s['TP']:<5} {s['FN']:<5} {s['total']:<7} {s['CDR']:<10}")

        # Step 3: Audit
        print("\n" + "-" * 50)
        print("[Step 3] LLM Context Auditor")
        audits = self.run_step3_audit(gold_triplets, step2)
        audit_stats = defaultdict(int)
        for a in audits:
            audit_stats[a["audit_action"]] += 1
        for action, count in audit_stats.items():
            print(f"  {action}: {count}")

        # Step 4: Build KG
        print("\n" + "-" * 50)
        print("[Step 4] Knowledge Graph Construction")
        kg_stats = self.run_step4_build_kg(gold_triplets, audits)
        print(f"  Nodes: {kg_stats['total_nodes']}")
        print(f"  Edges: {kg_stats['total_edges']}")
        print(f"  Uncertain edges: {kg_stats['uncertain_edges']}")
        print(f"  Avg confidence: {kg_stats['avg_confidence']:.3f}")
        print(f"  Conflicts: {kg_stats['conflict_breakdown']}")

        # Step 5: Sample queries
        print("\n" + "-" * 50)
        print("[Step 5] GraphRAG Inference — Sample Queries")
        test_queries = ["DO", "water_temp", "EC", "AITC", "PPFD", "air_temp"]
        query_results = []
        for q in test_queries:
            result = self.inference.answer_query(q)
            query_results.append(result)
            print(f"\n  Query: {q}")
            print(f"  Sources: {result['sources']}")
            print(f"  Min confidence: {result.get('min_confidence', 'N/A')}")
            if result.get('warnings'):
                for w in result['warnings'][:2]:
                    print(f"  {w}")

        # Export
        print("\n" + "-" * 50)
        print("[Export] Neo4j Cypher export ready")
        cypher = self.kg.export_for_neo4j()

        # Compile full results
        full_results = {
            "timestamp": datetime.now().isoformat(),
            "pipeline_version": "v1.2",
            "gold_standard_count": len(gold_triplets),
            "injection_count": len(injections),
            "step2_fpr": fpr_result,
            "step2_cdr": cdr_summary,
            "step3_audit_stats": dict(audit_stats),
            "step4_kg_stats": kg_stats,
            "step5_query_results": query_results,
            "neo4j_cypher_lines": len(cypher.split('\n'))
        }

        return full_results


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    pipeline = SelfAuditingPipeline()

    gs_path = os.path.join(os.path.dirname(__file__), "triplets_gold_standard.json")
    inj_path = os.path.join(os.path.dirname(__file__), "injection_test.json")

    results = pipeline.run_full_pipeline(gs_path, inj_path)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "pipeline_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[SAVED] Results → {out_path}")

    # Save Neo4j export
    cypher_path = os.path.join(os.path.dirname(__file__), "neo4j_import.cypher")
    with open(cypher_path, 'w', encoding='utf-8') as f:
        f.write(pipeline.kg.export_for_neo4j())
    print(f"[SAVED] Neo4j Cypher → {cypher_path}")
