#!/usr/bin/env python3
"""
Build semantic-aligned CSVs for the three 0530recording source files.
Requires:
  - professional interpreter/中文原声对应译文版本/0530_transcriptions.json
  - source speeches/source_text_zh/0530recording_{1,2,3} source.md

Uses the same semantic DP alignment as semantic_align_csvs.py.
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import csv, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

BASE         = Path(__file__).parent
TRANSCRIPT   = BASE / "professional interpreter" / "中文原声对应译文版本" / "0530_transcriptions.json"
SRC_DIR      = BASE / "source speeches" / "source_text_zh"
OUT_DIR      = BASE / "source speeches" / "aligned_csvs"

PAIRS = [
    ("0530recording_1 source.md", "0530recording_1 source.wav"),
    ("0530recording_2 source.md", "0530recording_2 source.wav"),
    ("0530recording_3 source.md", "0530recording_3 source.wav"),
]

CSV_COLUMNS = [
    "Segment_ID", "Source_Chinese", "Target_English",
    "Source_Audio_Start", "Translation_Audio_Start", "Delay_Seconds",
    "Language_Quality", "Expressiveness", "perceived latency",
    "evaluator_id", "Evaluation_Date", "Comments", "Flag_Uncertain",
    "similarity_score",
]

LOW_CONF = 0.30
_model = None
_tokenizer = None


def load_model(name="xlm-roberta-large"):
    global _model, _tokenizer
    if _model is None:
        print(f"Loading {name} …")
        _tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        _model = AutoModel.from_pretrained(name, local_files_only=True)
        _model.eval()
        print("Model ready.\n")


def embed(texts, batch_size=32):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = _tokenizer(batch, padding=True, truncation=True,
                         max_length=256, return_tensors="pt")
        with torch.no_grad():
            out = _model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)


def monotone_dp(src_embs, tgt_embs, min_tgt=1):
    N, D = src_embs.shape
    M = tgt_embs.shape[0]
    prefix = np.zeros((M + 1, D), dtype=np.float32)
    for j in range(M):
        prefix[j + 1] = prefix[j] + tgt_embs[j]

    def window_sim(a, b, sv):
        if b <= a:
            return -1.0
        mv = (prefix[b] - prefix[a]) / (b - a)
        norm = np.linalg.norm(mv)
        return float(np.dot(mv / norm, sv)) if norm > 1e-9 else 0.0

    NEG = -1e18
    dp = np.full((N + 1, M + 1), NEG)
    bp = np.full((N + 1, M + 1), -1, dtype=np.int32)
    dp[0][0] = 0.0

    for i in range(1, N + 1):
        sv = src_embs[i - 1]
        for j in range(i * min_tgt, M - (N - i) * min_tgt + 1):
            for k in range((i - 1) * min_tgt, j - min_tgt + 1):
                if dp[i - 1][k] == NEG:
                    continue
                s = dp[i - 1][k] + window_sim(k, j, sv)
                if s > dp[i][j]:
                    dp[i][j] = s
                    bp[i][j] = k

    boundaries, j = [], M
    for i in range(N, 0, -1):
        k = bp[i][j]
        boundaries.append((k, j - 1))
        j = k
    boundaries.reverse()
    return boundaries


def read_lines(md_path):
    return [l.strip() for l in md_path.read_text(encoding="utf-8").splitlines() if l.strip()]


def process_pair(src_lines, whisper_segs, out_path):
    N, M = len(src_lines), len(whisper_segs)
    tgt_texts = [s["text"].strip() for s in whisper_segs]
    min_tgt = max(1, M // (N * 3))

    src_embs = embed(src_lines)
    tgt_embs = embed(tgt_texts)
    bounds   = monotone_dp(src_embs, tgt_embs, min_tgt)

    rows, n_low = [], 0
    for i, (src, (a, b)) in enumerate(zip(src_lines, bounds), 1):
        grp  = whisper_segs[a: b + 1]
        text = " ".join(s["text"].strip() for s in grp if s["text"].strip())
        t_start = grp[0]["start"] if grp else ""

        if grp:
            ge = tgt_embs[a: b + 1].mean(axis=0)
            ge /= (np.linalg.norm(ge) + 1e-9)
            sim = float(np.dot(src_embs[i - 1], ge))
        else:
            sim = 0.0

        flag = "LOW_CONF" if sim < LOW_CONF else ""
        if flag:
            n_low += 1

        row = {c: "" for c in CSV_COLUMNS}
        row.update({
            "Segment_ID":              i,
            "Source_Chinese":          src,
            "Target_English":          text,
            "Translation_Audio_Start": t_start,
            "similarity_score":        f"{sim:.3f}" + (f" [{flag}]" if flag else ""),
        })
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerows(rows)

    avg = np.mean([float(r["similarity_score"].split()[0]) for r in rows])
    print(f"  ✓ {out_path.name}  ({N} src, {M} whisper)  avg_sim={avg:.3f}  low_conf={n_low}/{N}")


def main():
    if not TRANSCRIPT.exists():
        raise SystemExit(f"Transcript not found: {TRANSCRIPT}\nRun transcribe_0530.py first.")

    with TRANSCRIPT.open(encoding="utf-8") as f:
        data = json.load(f)
    idx = {e["file"]: e["segments"] for e in data}

    load_model()

    for md_name, audio_name in PAIRS:
        md_path = SRC_DIR / md_name
        if not md_path.exists():
            print(f"  ⚠ missing markdown: {md_path}")
            continue
        segs = idx.get(audio_name, [])
        if not segs:
            print(f"  ⚠ no transcript for {audio_name}")
            continue
        stem     = Path(md_name).stem.replace(" source", "")
        out_path = OUT_DIR / f"align_{stem}_zh-en.csv"
        src_lines = read_lines(md_path)
        process_pair(src_lines, segs, out_path)

    print(f"\nAll CSVs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
