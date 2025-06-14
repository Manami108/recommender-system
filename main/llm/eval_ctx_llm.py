#!/usr/bin/env python
# eval_ctx_llm.py (with one-to-one ground truth matching)
# ---------------------------------------------------------------------------
# Evaluate LLM contextual-prompt reranking over four recall sources:
#   1) full-text BM25
#   2) chunked BM25
#   3) full embedding
#   4) chunked embedding
# then rerank via contextual LLM
# Metrics updated to use one-to-one matching of recommendations to references.
# ---------------------------------------------------------------------------

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from neo4j import GraphDatabase

from chunking import clean_text, chunk_tokens
from recall import (
    recall_fulltext,
    recall_vector,
    fetch_metadata,
    embed
)
from rerank_llm import llm_contextual_rerank
from transformers import AutoTokenizer

# ─────────────────────────── CONFIG ──────────────────────────────────────── #

TESTSET_PATH  = Path("/home/abhi/Desktop/Manami/recommender-system/datasets/testset_2020_references.jsonl")
MAX_CASES     = 25
SIM_THRESHOLD = 0.90
TOPK_LIST     = (3, 5, 10, 15, 20)

NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "Manami1008")
driver     = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# Initialize tokenizer for chunking
TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# ─────────────────────────── METRIC HELPERS ─────────────────────────────── #

def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T

# ─────────────────────────── EVALUATION ─────────────────────────────────── #

def evaluate_case(paragraph: str,
                  true_pids: List[str],
                  target_year: Optional[int] = None) -> dict:
    # 1) Preprocess & chunk
    cleaned = clean_text(paragraph)
    chunks  = chunk_tokens(cleaned, TOKENIZER, win=128, stride=64)

    # 2) Recall from four sources
    bm25_full  = recall_fulltext(cleaned, k=40).assign(source="full_bm25")
    # chunk-level BM25: union across chunks
    bm25_rows = [recall_fulltext(ch, k=40) for ch in chunks]
    chunk_bm25 = (pd.concat(bm25_rows, ignore_index=True)
                    .drop_duplicates("pid")
                    .assign(source="chunk_bm25"))
    full_vec   = recall_vector(embed(cleaned), k=40, sim_threshold=0.30).assign(source="full_vec")
    vec_rows = [recall_vector(embed(ch), k=40, sim_threshold=0.30) for ch in chunks]
    chunk_vec  = (pd.concat(vec_rows, ignore_index=True)
                    .drop_duplicates("pid")
                    .assign(source="chunk_vec"))

    # 3) Combine & dedupe candidates
    pool = pd.concat([bm25_full, chunk_bm25, full_vec, chunk_vec], ignore_index=True)
    candidates = (pool
                  .sort_values(["source","sim"], ascending=[True, False])
                  .drop_duplicates("pid", keep="first")
                  .reset_index(drop=True))

    # 4) Metadata & filter by year
    meta   = fetch_metadata(candidates["pid"].tolist())
    merged = (candidates
              .merge(meta[["pid","abstract","year"]], on="pid", how="left")
              .dropna(subset=["abstract"]))
    if target_year is not None:
        merged = merged[merged["year"] <= target_year]
    if merged.empty:
        raise ValueError("No candidates after filtering")

    # 5) Contextual LLM rerank
    try:
        rer = llm_contextual_rerank(
            paragraph,
            merged[["pid","title","abstract"]],
            max_candidates=len(merged),
            batch_size=len(merged)
        )
        predicted = rer["pid"].tolist()
    except Exception:
        predicted = merged["pid"].tolist()

    # 6) Prepare reference embeddings
    ref_meta = fetch_metadata([str(p) for p in true_pids]).dropna(subset=["abstract"])
    ref_ids  = ref_meta["pid"].tolist()
    ref_embs = np.stack([embed(a) for a in ref_meta["abstract"]]) if ref_ids else np.zeros((0,768))

    # 7) Prepare candidate embeddings
    cand_ids  = predicted
    cand_absts= merged.set_index("pid").loc[cand_ids, "abstract"].tolist()
    cand_embs = np.stack([embed(a) for a in cand_absts])

    # 8) Compute similarity matrix
    sims = cosine_matrix(cand_embs, ref_embs) if ref_ids else np.zeros((len(cand_ids),0))

    # 9) One-to-one greedy matching
    unmatched = set(range(len(ref_ids)))
    hits = []
    for i in range(len(cand_ids)):
        if not unmatched:
            hits.append(False)
            continue
        # find best unmatched ref for this candidate
        ref_idxs = list(unmatched)
        sim_vals = sims[i, ref_idxs]
        best_j  = sim_vals.argmax()
        best_sim= sim_vals[best_j]
        if best_sim >= SIM_THRESHOLD:
            hits.append(True)
            unmatched.remove(ref_idxs[best_j])
        else:
            hits.append(False)

    # 10) Compute metrics
    results = {}
    n_rel    = len(ref_ids)
    for k in TOPK_LIST:
        topk = hits[:k]
        results[f"P@{k}"]  = sum(topk)/k
        results[f"HR@{k}"] = float(any(topk))
        covered = sum(topk)
        results[f"R@{k}"]  = covered / n_rel if n_rel else 0.0
        # NDCG
        dcg  = sum(r/np.log2(i+2) for i,r in enumerate(topk))
        idcg = sum(1/np.log2(i+2) for i in range(min(n_rel,k)))
        results[f"NDCG@{k}"] = dcg/idcg if idcg>0 else 0.0

    return results


def main():
    df = pd.read_json(TESTSET_PATH, lines=True).head(MAX_CASES)
    all_m = []
    for case in df.to_dict("records"):
        yr = case.get("year") or None
        all_m.append(evaluate_case(case["paragraph"], case.get("references", []), yr))

    avg = pd.DataFrame(all_m).mean(numeric_only=True)
    print("\nContextual LLM rerank average metrics over four recall sources:\n")
    print(avg)

if __name__ == "__main__":
    main()
