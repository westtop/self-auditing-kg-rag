# -*- coding: utf-8 -*-
"""Self-Auditing extraction pipeline wired to the local HF backend, for the
strawberry validation experiment. Extraction is per-paper and cacheable; the
rest is pure-python aggregation so subsampling needs no extra LLM calls."""
import os, re, json
from strawberry_config import STRAWBERRY_NORMAL_RANGE as NR, UNIT_ALIASES
from metrics import make_param_key

param_key = make_param_key(NR)
NUM_RE = re.compile(r"\d+\.?\d*")

# ---------- unit normalization ----------
def canonical_unit(unit):
    u = (unit or "").strip().lower()
    for canon, aliases in UNIT_ALIASES.items():
        if u == canon.lower() or u in [a.lower() for a in aliases]:
            return canon
    return (unit or "").strip()

def parse_values(obj):
    s = (obj or "").replace("~", "-").replace("/", " ").replace(" to ", "-")
    s = re.sub(r"(?<=\d),(?=\d)", "", s)          # remove thousands separators (25,000 -> 25000)
    out = []
    for tok in re.split(r"[\s,]+", s):
        if not tok or re.search(r"[a-zA-Z%°]", tok):
            continue
        out += [float(x) for x in NUM_RE.findall(tok)]
    return out

def normalize_triplet(t):
    t = dict(t)
    t["unit_canon"] = canonical_unit(t.get("unit", ""))
    vals = parse_values(t.get("object", ""))
    raw = (t.get("unit", "") or "").lower()
    if "temp" in t.get("subject", "").lower() or t["unit_canon"] == "C":
        if "f" in raw and "c" not in raw:
            vals = [(v - 32) * 5 / 9 for v in vals]
        elif raw.strip() in ("k", "kelvin"):
            vals = [v - 273.15 for v in vals]
    t["values"] = vals
    return t

# ---------- PDF text ----------
def read_any_text(path, max_chars=None):
    """Read .pdf (via pypdf) or .txt (plain) into normalized text."""
    p = str(path)
    if p.lower().endswith(".txt"):
        try:
            text = open(p, encoding="utf-8", errors="ignore").read()
        except Exception as e:
            print(f"  [warn] {os.path.basename(p)}: {e}"); text = ""
        text = re.sub(r"[ \t]+", " ", text)
        return text[:max_chars] if max_chars else text
    return extract_pdf_text(path, max_chars=max_chars)

def extract_pdf_text(path, max_chars=None):
    text = ""
    try:
        from pypdf import PdfReader
        for pg in PdfReader(str(path)).pages:
            text += (pg.extract_text() or "") + "\n"
    except Exception as e:
        print(f"  [warn] {os.path.basename(str(path))}: {e}")
    text = re.sub(r"[ \t]+", " ", text)
    return text[:max_chars] if max_chars else text

# ---------- Step 1: extraction (per paper, cache) ----------
EXTRACTION_SYSTEM = """You are an expert agricultural knowledge extraction system.
Extract quantitative cultivation-parameter triplets from the given scientific text about
strawberry hydroponics/soilless culture. Return ONLY triplets that contain a measurable
parameter (water/root-zone temperature, air temperature, EC, pH, dissolved oxygen (DO),
PPFD, photoperiod, DLI, etc.).
Output strict JSON: {"triplets":[{"subject":..., "predicate":..., "object":<value with unit>,
"unit":..., "condition":<experimental condition or "">}]}
Rules: keep numeric values verbatim; include the condition when stated; do NOT invent values."""

def extract_paper(text, source_id, llm, max_chars=12000):
    data = llm.chat_json(EXTRACTION_SYSTEM, f"SOURCE_ID={source_id}\n\nTEXT:\n{text[:max_chars]}")
    out = []
    for t in (data.get("triplets", []) if isinstance(data, dict) else []):
        out.append({"subject": str(t.get("subject", "")).strip(),
                    "predicate": str(t.get("predicate", "")).strip(),
                    "object": str(t.get("object", "")).strip(),
                    "unit": str(t.get("unit", "")).strip(),
                    "condition": str(t.get("condition", "")).strip(),
                    "source_id": source_id})
    return out

def extract_paper_chunked(text, source_id, llm, chunk_size=8000, overlap=400):
    """Extract from the WHOLE paper via overlapping windows, then merge+dedup.
    Captures Methods/Results deep in the paper that a single 12k window misses."""
    if not text:
        return []
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size]); i += chunk_size - overlap
    seen, merged = set(), []
    for ch in chunks:
        for t in extract_paper(ch, source_id, llm, max_chars=chunk_size):
            key = (t["subject"].lower(), t["predicate"].lower(), t["object"].lower())
            if key in seen:
                continue
            seen.add(key); merged.append(t)
    return merged

def extract_corpus(pdf_dir, llm, cache_path="extractions.json", max_chars=12000,
                   chunked=True, chunk_size=8000):
    """Extract all S*.pdf under pdf_dir, caching per-paper results. Re-run is free (uses cache)."""
    from pathlib import Path
    cache = {}
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path, encoding="utf-8"))
    cand = list(Path(pdf_dir).glob("*.pdf")) + list(Path(pdf_dir).glob("*.txt"))
    def _num(name):
        m = re.match(r"^(S?\d+)", name)
        return int(re.match(r"^S?(\d+)", name).group(1)) if m else None
    files = sorted([p for p in cand if _num(p.name) is not None], key=lambda p: _num(p.name))
    for p in files:
        m = re.match(r"^(S?\d+)", p.name); sid = m.group(1)
        if sid in cache:
            continue
        txt = read_any_text(p, max_chars=None)
        try:
            if chunked:
                cache[sid] = extract_paper_chunked(txt, sid, llm, chunk_size=chunk_size)
            else:
                cache[sid] = extract_paper(txt, sid, llm, max_chars=max_chars)
        except Exception as e:
            print(f"  [warn] extract {sid}: {e}"); cache[sid] = []
        json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"  {sid}: {len(cache[sid])} triplets")
    return cache   # {source_id: [triplets]}

# ---------- Step 2: rule detector (cross-source + range) ----------
def rule_detector(triplets):
    groups = {}
    for i, t in enumerate(triplets):
        pk = param_key(t.get("subject", ""))
        if pk: groups.setdefault(pk, []).append(i)
    flags = [None] * len(triplets)
    for pk, idxs in groups.items():
        if len({triplets[i].get("object", "").strip() for i in idxs}) > 1:
            for i in idxs: flags[i] = "T1"
    for i, t in enumerate(triplets):
        pk = param_key(t.get("subject", ""))
        if pk and t.get("values"):
            lo, hi, _ = NR[pk]
            if any(v < lo or v > hi for v in t["values"]): flags[i] = "T1"
    for i, t in enumerate(triplets):
        if flags[i] is None and not t.get("condition", "").strip() and param_key(t.get("subject", "")):
            flags[i] = "T3_candidate"
    return flags

# ---------- Step 3: LLM audit (optional; used on full corpus) ----------
AUDIT_SYSTEM = """You are a scientific fact-checking auditor for strawberry hydroponics.
Given a flagged triplet, decide TRUE_CONFLICT vs FALSE_ALARM (a cross-source difference is a
FALSE_ALARM if explainable by cultivar/stage/substrate/season). Return JSON
{"verdict":"TRUE_CONFLICT"|"FALSE_ALARM"}."""

def llm_audit(triplet, flag, ctx, llm):
    d = llm.chat_json(AUDIT_SYSTEM, f"Flag: {flag}\nTriplet: {json.dumps(triplet, ensure_ascii=False)}\n"
                                    f"Context:\n{ctx[:1200]}")
    return d.get("verdict", "FALSE_ALARM") if isinstance(d, dict) else "FALSE_ALARM"

def assign_confidence(flag, verdict):
    if flag is None: return "Verified"
    if flag == "T3_candidate": return "Disputed" if verdict == "TRUE_CONFLICT" else "Verified"
    if flag in ("T1", "T2"): return "Conflicted" if verdict == "TRUE_CONFLICT" else "Verified"
    return "Verified"

def audit_corpus(triplets, llm=None, docs=None, do_audit=False):
    """Normalize + rule-detect (+optional LLM audit) -> audited triplets with confidence.
    For the convergence metric, do_audit=False is fine (robust median handles outliers)."""
    norm = [normalize_triplet(t) for t in triplets]
    flags = rule_detector(norm)
    audited = []
    for i, t in enumerate(norm):
        verdict = None
        if do_audit and llm is not None and flags[i] in ("T1", "T2", "T3_candidate"):
            ctx = (docs or {}).get(t["source_id"], "")
            try: verdict = llm_audit(t, flags[i], ctx, llm)
            except Exception: verdict = "FALSE_ALARM"
        audited.append(dict(t, flag=flags[i], verdict=verdict,
                            confidence=assign_confidence(flags[i], verdict)))
    return audited
