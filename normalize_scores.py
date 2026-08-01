#!/usr/bin/env python3
"""
Per-rater Z-score normalization to remove systematic rater bias.

Applies to both:
  - Existing train/dev/test sets (raters R01, R02, R04)
  - professional_set.json (raters 胡蝶, 张凤)

Steps:
  1. Compute per-rater (mean, std) for LQ and EXP
  2. Z-normalize each score: z = (x - mu_r) / sigma_r
  3. Re-scale to global target distribution: x_norm = z * sigma_global + mu_global
  4. Clip to [0, 3]
  5. Save normalized versions of all files

Outputs:
  train_set_norm.json, dev_set_norm.json, test_set_norm.json
  professional_set_norm.json
  normalization_stats.json  (per-rater parameters + global target)
"""

import json
import statistics
import math
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).parent


# ---------------------------------------------------------------------------
# Load all data and collect per-rater raw scores
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


existing_files = {
    "train": BASE / "train_set.json",
    "dev":   BASE / "dev_set.json",
    "test":  BASE / "test_set.json",
}
prof_file = BASE / "professional_set.json"

# Collect raw scores per rater
rater_lq:  dict[str, list[float]] = defaultdict(list)
rater_exp: dict[str, list[float]] = defaultdict(list)

for split, path in existing_files.items():
    for item in load_json(path):
        raters = item.get("raters", [])
        lq     = item.get("LQ")
        exp    = item.get("EXP")
        for r in raters:
            if lq  is not None: rater_lq[r].append(lq)
            if exp is not None: rater_exp[r].append(exp)

for item in load_json(prof_file):
    raters = item.get("raters", [])
    lq     = item.get("LQ")
    exp    = item.get("EXP")
    for r in raters:
        if lq  is not None: rater_lq[r].append(lq)
        if exp is not None: rater_exp[r].append(exp)


# ---------------------------------------------------------------------------
# Compute per-rater parameters
# ---------------------------------------------------------------------------

def rater_params(rater_scores: dict[str, list[float]]) -> dict[str, dict]:
    params = {}
    for r, vals in rater_scores.items():
        params[r] = {
            "mean": statistics.mean(vals),
            "std":  statistics.stdev(vals) if len(vals) > 1 else 1.0,
            "n":    len(vals),
        }
    return params

lq_params  = rater_params(rater_lq)
exp_params = rater_params(rater_exp)

# Global target: pooled mean and std across all raters
all_lq  = [v for vals in rater_lq.values()  for v in vals]
all_exp = [v for vals in rater_exp.values() for v in vals]

lq_global  = {"mean": statistics.mean(all_lq),  "std": statistics.stdev(all_lq)}
exp_global = {"mean": statistics.mean(all_exp), "std": statistics.stdev(all_exp)}

print("Per-rater LQ parameters:")
for r, p in sorted(lq_params.items()):
    print(f"  {r}: mean={p['mean']:.3f}  std={p['std']:.3f}  n={p['n']}")
print(f"  → Global target: mean={lq_global['mean']:.3f}  std={lq_global['std']:.3f}")

print("\nPer-rater EXP parameters:")
for r, p in sorted(exp_params.items()):
    print(f"  {r}: mean={p['mean']:.3f}  std={p['std']:.3f}  n={p['n']}")
print(f"  → Global target: mean={exp_global['mean']:.3f}  std={exp_global['std']:.3f}")


# ---------------------------------------------------------------------------
# Normalization function
# ---------------------------------------------------------------------------

def normalize(raw: float, rater: str, field: str) -> float:
    """Z-normalize a score for a given rater, then re-scale to global target."""
    if field == "LQ":
        params, global_p = lq_params, lq_global
    else:
        params, global_p = exp_params, exp_global

    if rater not in params:
        return raw  # unknown rater: pass through

    mu_r = params[rater]["mean"]
    sd_r = params[rater]["std"]
    if sd_r < 1e-6:
        return raw  # degenerate rater (zero variance)

    z     = (raw - mu_r) / sd_r
    renorm = z * global_p["std"] + global_p["mean"]
    return round(max(0.0, min(3.0, renorm)), 3)


def normalize_item(item: dict) -> dict:
    """Return a copy of item with LQ and EXP normalized."""
    item = dict(item)
    raters = item.get("raters", [])
    if not raters:
        return item

    # For items with multiple raters, average the normalized scores
    lq_raw  = item.get("LQ")
    exp_raw = item.get("EXP")

    # We normalize the *averaged* raw score using the rater group average parameters
    # Build a synthetic "rater pool" key for mixed raters
    if lq_raw is not None:
        # Average of per-rater normalizations
        lq_norms = [normalize(lq_raw, r, "LQ") for r in raters if r in lq_params]
        if lq_norms:
            item["LQ_norm"]  = round(sum(lq_norms) / len(lq_norms), 3)
            item["LQ_raw"]   = lq_raw

    if exp_raw is not None:
        exp_norms = [normalize(exp_raw, r, "EXP") for r in raters if r in exp_params]
        if exp_norms:
            item["EXP_norm"] = round(sum(exp_norms) / len(exp_norms), 3)
            item["EXP_raw"]  = exp_raw

    return item


# ---------------------------------------------------------------------------
# Apply normalization and save
# ---------------------------------------------------------------------------

def process_and_save(data, out_path):
    normed = [normalize_item(item) for item in data]
    out_path.write_text(json.dumps(normed, ensure_ascii=False, indent=2), encoding="utf-8")

    # Report change statistics
    lq_changes, exp_changes = [], []
    for item in normed:
        if "LQ_raw" in item and "LQ_norm" in item:
            lq_changes.append(abs(item["LQ_norm"] - item["LQ_raw"]))
        if "EXP_raw" in item and "EXP_norm" in item:
            exp_changes.append(abs(item["EXP_norm"] - item["EXP_raw"]))

    lq_mean_chg  = statistics.mean(lq_changes)  if lq_changes  else 0
    exp_mean_chg = statistics.mean(exp_changes) if exp_changes else 0
    print(f"  {out_path.name}: LQ avg_change={lq_mean_chg:.3f}  EXP avg_change={exp_mean_chg:.3f}")

print("\nNormalizing and saving:")
for split, path in existing_files.items():
    data = load_json(path)
    out  = BASE / f"{path.stem}_norm.json"
    process_and_save(data, out)

data = load_json(prof_file)
process_and_save(data, BASE / "professional_set_norm.json")

# ---------------------------------------------------------------------------
# Save normalization parameters for reproducibility
# ---------------------------------------------------------------------------

stats = {
    "LQ": {
        "per_rater": lq_params,
        "global_target": lq_global,
    },
    "EXP": {
        "per_rater": exp_params,
        "global_target": exp_global,
    },
}
(BASE / "normalization_stats.json").write_text(
    json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
)

# ---------------------------------------------------------------------------
# Post-normalization inter-rater agreement check
# ---------------------------------------------------------------------------
print("\n=== Before vs After normalization (胡蝶 vs 张凤 on shared items) ===")

import chardet, csv
from scipy.stats import pearsonr

EVAL_DIR = BASE / "professional interpreter" / "evaluation"

def detect_and_read(fpath):
    raw  = fpath.read_bytes()
    enc  = chardet.detect(raw)["encoding"] or "utf-8"
    text = raw.decode(enc, errors="replace")
    lines = text.splitlines()
    start = 0 if (lines and "Segment_ID" in lines[0]) else 1
    return [r for r in csv.DictReader(lines[start:]) if r.get("Segment_ID","").strip()]

def safe_float(v):
    try: return float((v or "").strip())
    except: return None

by_key = defaultdict(dict)
for f in EVAL_DIR.glob("**/*.csv"):
    if f.name.startswith("."): continue
    folder = f.parent.name
    rater  = "胡蝶" if "胡蝶" in folder else ("张凤" if "张凤" in folder else None)
    if not rater: continue
    file_id = f.stem.split("_")[0]
    for row in detect_and_read(f):
        seg = row.get("Segment_ID","").strip()
        key = f"{file_id}_{seg}"
        by_key[key][rater] = {
            "LQ":  safe_float(row.get("Language_Quality")),
            "EXP": safe_float(row.get("Expressiveness")),
        }

both = {k: v for k, v in by_key.items() if "胡蝶" in v and "张凤" in v}

for field in ["LQ", "EXP"]:
    raw_a, raw_b, norm_a, norm_b = [], [], [], []
    for v in both.values():
        a = v["胡蝶"][field]
        b = v["张凤"][field]
        if a is None or b is None: continue
        raw_a.append(a);  raw_b.append(b)
        norm_a.append(normalize(a, "胡蝶", field))
        norm_b.append(normalize(b, "张凤", field))

    r_raw,  _ = pearsonr(raw_a,  raw_b)
    r_norm, _ = pearsonr(norm_a, norm_b)
    mae_raw   = sum(abs(a-b) for a,b in zip(raw_a,  raw_b))  / len(raw_a)
    mae_norm  = sum(abs(a-b) for a,b in zip(norm_a, norm_b)) / len(norm_a)

    print(f"  {field}: Pearson  raw={r_raw:.3f} → norm={r_norm:.3f}")
    print(f"       MAE      raw={mae_raw:.3f} → norm={mae_norm:.3f}")

print(f"\nAll files saved. Normalization params → normalization_stats.json")
