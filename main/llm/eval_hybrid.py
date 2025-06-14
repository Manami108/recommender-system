#!/usr/bin/env python
# eval_hybrid.py (with one-to-one ground truth matching)
# ---------------------------------------------------------------------------
# Evaluate hybrid BM25+embedding per-chunk (union) with one-to-one matching
# ---------------------------------------------------------------------------

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from neo4j import GraphDatabase

from chunking import clean_text, chunk_tokens
from recall import recall_by_chunks, fetch_metadata, embed

# ─────────────────────────── CONFIG ──────────────────────────────────────── #
TESTSET_PATH  = Path("/home/abhi/Desktop/Manami/recommender-system/datasets/testset_2020_references.jsonl")
MAX_CASES     = 25
SIM_THRESHOLD = 0.90
TOPK_LIST     = (3, 5, 10, 15, 20)

NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "Manami1008")
driver     = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# Initialize tokenizer once
from transformers import AutoTokenizer
TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# ─────────────────────────── METRIC HELPERS ────────────────────────────────── #
def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T

# ─────────────────────────── EVALUATION ─────────────────────────────────── #
def evaluate_case(paragraph: str, true_pids: List[str], target_year: Optional[int]):
    # 1) Clean & chunk
    cleaned = clean_text(paragraph)
    chunks  = chunk_tokens(cleaned, TOKENIZER, win=128, stride=64)

    # 2) Hybrid recall per chunk
    hybrid = recall_by_chunks(chunks, k_bm25=40, k_vec=40, sim_th=0.30)

    # 3) Metadata & filter
    meta   = fetch_metadata(hybrid["pid"].tolist())
    merged = hybrid.merge(meta[["pid","abstract","year"]], on="pid").dropna(subset=["abstract"])
    if target_year is not None:
        merged = merged[merged["year"] <= target_year]

    # 4) Prepare references
    ref_meta = fetch_metadata([str(p) for p in true_pids]).dropna(subset=["abstract"])
    ref_ids  = ref_meta["pid"].tolist()
    ref_embs = np.stack([embed(a) for a in ref_meta["abstract"]]) if ref_ids else np.zeros((0,768))

    # 5) Embed candidates and compute sims
    cand_ids   = merged["pid"].tolist()
    cand_texts = merged["abstract"].tolist()
    cand_embs  = np.stack([embed(a) for a in cand_texts])
    sims       = cosine_matrix(cand_embs, ref_embs) if ref_ids else np.zeros((len(cand_ids),0))

    # 6) One-to-one greedy matching
    unmatched = set(range(len(ref_ids)))
    hits      = []
    for i in range(len(cand_ids)):
        if not unmatched:
            hits.append(False)
            continue
        ref_idxs = list(unmatched)
        sim_vals = sims[i, ref_idxs]
        best_j   = sim_vals.argmax()
        best_sim = sim_vals[best_j]
        if best_sim >= SIM_THRESHOLD:
            hits.append(True)
            unmatched.remove(ref_idxs[best_j])
        else:
            hits.append(False)

    # 7) Compute metrics
    results = {}
    n_rel    = len(ref_ids)
    for k in TOPK_LIST:
        topk    = hits[:k]
        p_at_k   = sum(topk) / k
        hr_at_k  = float(any(topk))
        covered  = sum(topk)
        r_at_k   = covered / n_rel if n_rel else 0.0
        # NDCG
        dcg  = sum(r/np.log2(i+2) for i, r in enumerate(topk))
        idcg = sum(1/np.log2(i+2) for i in range(min(n_rel, k)))
        ndcg = dcg/idcg if idcg > 0 else 0.0
        results.update({
            f"P@{k}": p_at_k,
            f"HR@{k}": hr_at_k,
            f"R@{k}": r_at_k,
            f"NDCG@{k}": ndcg
        })
    return results


def main():
    df = pd.read_json(TESTSET_PATH, lines=True).head(MAX_CASES)
    all_m = []
    for case in df.to_dict("records"):
        yr = case.get("year") or None
        all_m.append(evaluate_case(case["paragraph"], case.get("references", []), yr))
    avg = pd.DataFrame(all_m).mean(numeric_only=True)
    print("\nHybrid BM25+vector per-chunk average metrics:\n")
    print(avg)

if __name__ == "__main__":
    main()
