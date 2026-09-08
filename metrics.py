# -*- coding: utf-8 -*-
"""Primary metric = parameter-value convergence; plus variance across subsamples.
Pure-python (no LLM/GPU). Works on audited triplets from the pipeline."""
import numpy as np
import re


def _norm_unit(u):
    return re.sub(r"[\s\u2219\u00b7\u207b\u00b2\u00b3\u2013\u2212\-\./]", "", (u or "").lower())

def unit_compatible(param, raw_unit):
    """Reject values whose unit does not match the parameter (e.g. lux/%/days/MJ mislabeled as PPFD).
    This is the T2 (unit-mismatch) guard for aggregation."""
    u = _norm_unit(raw_unit); raw = raw_unit or ""
    if param == "PPFD":
        if not u: return False
        if "lux" in u or u.endswith("lx") or u == "lx": return False
        if "%" in raw or "day" in u or "mj" in u or "wh" in u: return False
        return "mol" in u
    if param == "DLI":
        return ("mol" in u and "d" in u) or "mj" in u
    if param in ("water_temp", "air_temp"):
        return u in ("", "c", "f", "k") or "c" in u or "f" in u or "k" in u
    if param == "EC":
        return "s" in u
    if param == "DO":
        return "mg" in u or "ppm" in u
    if param == "pH":
        return u in ("", "ph", "phunit")
    if param == "photoperiod":
        return u in ("", "h", "hr", "hrs", "hour", "hours")
    return True


CONF_WEIGHT = {"Verified": 1.0, "Disputed": 0.5, "Conflicted": 0.25}

def make_param_key(normal_range):
    """Return a subject->param mapper for a given reference table (case/substring safe)."""
    def pk(subject):
        s = (subject or "").strip().lower().replace(" ", "_")
        if "temp" in s:
            return "air_temp" if "air" in s else "water_temp"
        table = {"do":"DO","dissolved_oxygen":"DO","ec":"EC","electrical_conductivity":"EC",
                 "ph":"pH","ppfd":"PPFD","light_intensity":"PPFD","photoperiod":"photoperiod",
                 "daylength":"photoperiod","dli":"DLI"}
        if s in table and table[s] in normal_range: return table[s]
        for a,k in table.items():
            if a in s.split("_") and k in normal_range: return k
        for k in normal_range:
            if k.lower()==s: return k
        return None
    return pk

def weighted_median(values, weights):
    if not values: return None
    order = np.argsort(values)
    v = np.array(values)[order]; w = np.array(weights)[order]
    cw = np.cumsum(w); cutoff = w.sum()/2.0
    return float(v[np.searchsorted(cw, cutoff)])

def derive_param_values(audited, normal_range):
    """Collapse a corpus of audited triplets into one representative value per parameter
    (confidence-weighted median). `audited` items: {subject, values:[...], confidence}."""
    pk = make_param_key(normal_range)
    bucket = {p: ([], []) for p in normal_range}
    for t in audited:
        p = pk(t.get("subject",""))
        if p is None or not t.get("values"): continue
        if not unit_compatible(p, t.get("unit","")): continue   # T2 unit-mismatch guard
        lo, hi, _ = normal_range[p]
        w = CONF_WEIGHT.get(t.get("confidence","Verified"), 1.0)
        for v in t["values"]:
            if lo-0.5*(hi-lo) <= v <= hi+0.5*(hi-lo):   # drop gross outliers
                bucket[p][0].append(v); bucket[p][1].append(w)
    return {p: weighted_median(vals, wts) for p,(vals,wts) in bucket.items() if vals}

def convergence_error(derived, reference, normal_range):
    """Normalized abs error per param: |derived - reference| / span.  Lower = better."""
    out = {}
    for p, dval in derived.items():
        if reference.get(p) is None: continue
        lo, hi, _ = normal_range[p]; span = (hi-lo) or 1.0
        out[p] = abs(dval - reference[p]) / span
    return out

def variance_summary(subset_derived_list, normal_range):
    """Across N subsample runs, per-param mean/std/95%CI of the derived value."""
    params = normal_range.keys()
    summ = {}
    for p in params:
        vals = [d[p] for d in subset_derived_list if d.get(p) is not None]
        if len(vals) < 2: continue
        a = np.array(vals)
        summ[p] = {"n": len(a), "mean": float(a.mean()), "std": float(a.std(ddof=1)),
                   "ci95": (float(np.percentile(a,2.5)), float(np.percentile(a,97.5)))}
    return summ

if __name__ == "__main__":
    from strawberry_config import STRAWBERRY_NORMAL_RANGE as NR, STRAWBERRY_CONSENSUS as CON
    rng = np.random.default_rng(0)
    # synthetic "full corpus": many triplets clustered near consensus
    def synth(n, jitter):
        out=[]
        for p,(lo,hi,u) in NR.items():
            c=CON[p]
            for _ in range(n):
                out.append({"subject":p,"values":[float(rng.normal(c, jitter*(hi-lo)))],
                            "confidence":"Verified"})
        return out
    full = derive_param_values(synth(40,0.08), NR)
    subs = [derive_param_values(synth(6,0.12), NR) for _ in range(25)]   # 25 sparse draws
    print("Full-corpus derived:", {k:round(v,2) for k,v in full.items()})
    errs_vs_full = [convergence_error(s, full, NR) for s in subs]
    errs_vs_con  = [convergence_error(s, CON,  NR) for s in subs]
    def meanerr(el): 
        allp={p:np.mean([e[p] for e in el if p in e]) for p in NR}
        return {k:round(v,3) for k,v in allp.items()}
    print("Mean norm-error vs FULL  :", meanerr(errs_vs_full))
    print("Mean norm-error vs CONSENSUS:", meanerr(errs_vs_con))
    vs=variance_summary(subs, NR)
    print("Variance (EC):", {k:round(v,3) if not isinstance(v,tuple) else tuple(round(x,2) for x in v) for k,v in vs["EC"].items()})
