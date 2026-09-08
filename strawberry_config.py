# -*- coding: utf-8 -*-
"""Strawberry hydroponics — external ground-truth reference tables for the
data-rich validation experiment (parallel to the wasabi Appendix A tables).

Values are literature-consensus ranges for hydroponic/soilless strawberry.
NOTE: replace the placeholder citations with peer-reviewed sources during the
formal lit review (same curation policy as the wasabi Normal Range Table)."""

# --- Ground-truth Normal Range Table : param -> (low, high, unit) ---
# Consensus ranges compiled from horticultural references (see SOURCES below).
STRAWBERRY_NORMAL_RANGE = {
    # re-sourced from literature; citations & confidence in strawberry_consensus_sources.md
    "water_temp":   (18.0, 22.0, "C"),            # root/solution; photosynthesis optimum ~20C [Moderate]
    "air_temp":     (15.0, 25.0, "C"),            # stage-dependent; 24h-avg ~20C [Moderate]
    "DO":           (5.0,  8.0,  "mg/L"),         # UNCERTAIN: no established strawberry DO threshold [S3]
    "EC":           (0.9,  1.8,  "dS/m"),         # yield-opt ~0.9-1.0 [S2]; quality/commercial higher
    "pH":           (5.5,  6.5,  ""),             # nutrient-management reviews [S5]
    "PPFD":         (100.0, 500.0, "umol/m2/s"),  # sole-source tested 200-450 [S1]
    "photoperiod":  (12.0, 16.0, "h"),            # 16h optimal for fruiting [S1]
    "DLI":          (12.0, 26.0, "mol/m2/d"),     # derived PPFD x photoperiod [S1]; extension 20-25 [S6]
}

# --- External literature REFERENCE value (NOT ground truth) (single "textbook answer") per parameter, for the
#     convergence metric (how close does a sparse subset land to the known value). ---
STRAWBERRY_REFERENCE = {
    # Independently-reported literature reference value per parameter (NOT an absolute
    # "answer"): cultivation optima are context-dependent. See strawberry_consensus_sources.md.
    "water_temp":  20.0,   # C          [Moderate-High, S7]
    "air_temp":    20.0,   # C          [Moderate]
    "DO":          6.0,    # mg/L       [Uncertain: no established threshold, S3]
    "EC":          1.2,    # dS/m       [Moderate, objective-dependent; yield-opt ~0.9, S2]
    "pH":          6.0,    #            [Moderate, S5]
    "PPFD":        300.0,  # umol/m2/s  [High, S1: PMC9965992]
    "photoperiod": 16.0,   # h          [High, S1: PMC9965992]
    "DLI":         20.0,   # mol/m2/d   [Low-Moderate, S1/S6/S8; fruiting DLI weakly sourced]
}

# Reuse the domain-agnostic unit alias table from the main pipeline
UNIT_ALIASES = {
    "C":         ["c", "celsius", "°c", "degc", "deg c"],
    "mg/L":      ["mg/l", "ppm", "mg l-1", "mgl"],
    "dS/m":      ["ds/m", "ms/cm", "us/cm", "µs/cm", "μs/cm", "dsm"],
    "":          ["ph unit", "ph", "dimensionless", "-"],
    "umol/m2/s": ["µmol/m²/s", "μmol/m2/s", "umol m-2 s-1", "µe/m²/s", "umol/m^2/s"],
    "h":         ["hours", "hr", "hrs", "hour"],
    "mol/m2/d":  ["mol/m^2/d", "mol m-2 d-1", "dli"],
}

# Provenance for the ranges above (to be upgraded to peer-reviewed citations).
SOURCES = {
    "detail": "See strawberry_consensus_sources.md for per-parameter citations, quoted findings, "
              "confidence levels, and verification notes.",
    "S1": "PMC9965992 Strawberry 'Albion' sole-source lighting (PPFD/photoperiod). NOTE: also corpus #39.",
    "S2": "ResearchGate 262541629 Nutrient solution concentration on strawberry (EC vs yield/quality).",
    "S3": "Frontiers Plant Sci 2026 doi:10.3389/fpls.2026.1829367 (DO threshold ABSENT). corpus #3.",
    "S5": "OSU Indoor Berry / UMN Extension nutrient management (pH).",
    "S6": "OSU Indoor Berry environment/lighting (DLI, extension).",
    "S7": "CCSE J. Agric. Sci. Root-Zone Temperature on hydroponic strawberry (18C uptake / 20C growth).",
    "S8": "IJABE runner propagation LED (DLI 11.5-17.3, propagation stage only).",
}

STRAWBERRY_CONSENSUS = STRAWBERRY_REFERENCE   # backward-compat alias

if __name__ == "__main__":
    print("Strawberry ground-truth parameters:", len(STRAWBERRY_NORMAL_RANGE))
    for k,(lo,hi,u) in STRAWBERRY_NORMAL_RANGE.items():
        print(f"  {k:12s}: {lo}~{hi} {u}   (ref {STRAWBERRY_REFERENCE[k]})")
