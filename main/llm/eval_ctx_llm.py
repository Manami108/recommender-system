#!/usr/bin/env python
# eval_ctx_llm.py (fixed chunk-only recall)
# ---------------------------------------------------------------------------
# Evaluate LLM contextual-prompt reranking over four recall sources:
#   1) full-text BM25
#   2) chunked BM25
#   3) full embedding
#   4) chunked embedding
# then rerank via contextual LLM
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

TESTSET_PATH  = Path("/home/abhi/Desktop/Manami/recommender-system/datasets/testset_300_references.jsonl")
MAX_CASES     = 2
SIM_THRESHOLD = 0.95
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

def precision_by_sim(pred, sim_map, k):
    return sum(sim_map.get(p, False) for p in pred[:k]) / k if k else 0.0

def recall_by_sim(pred, sim_map, k, n_rel):
    return sum(sim_map.get(p, False) for p in pred[:k]) / n_rel if n_rel else 0.0

def hit_rate_by_sim(pred, sim_map, k):
    return float(any(sim_map.get(p, False) for p in pred[:k]))

def ndcg_by_sim(pred, sim_map, k):
    rels = [1 if sim_map.get(p, False) else 0 for p in pred[:k]]
    dcg  = sum(r/np.log2(i+2) for i, r in enumerate(rels))
    ideal = sorted(rels, reverse=True)
    idcg  = sum(r/np.log2(i+2) for i, r in enumerate(ideal))
    return dcg/idcg if idcg>0 else 0.0

def fetch_year_by_doi(doi: Optional[str]) -> Optional[int]:
    if not doi:
        return None
    try:
        q = "MATCH (p:Paper {doi:$doi}) RETURN p.year AS year"
        with driver.session() as sess:
            rec = sess.run(q, doi=doi).single()
        return rec.get("year") if rec and rec.get("year") is not None else None
    except:
        return None

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
    bm25_rows = []
    for ch in chunks:
        bm25_rows.append(recall_fulltext(ch, k=40))
    chunk_bm25 = (
        pd.concat(bm25_rows, ignore_index=True)
          .drop_duplicates("pid")
          .assign(source="chunk_bm25")
    )
    full_vec   = recall_vector(embed(cleaned), k=40, sim_threshold=0.30).assign(source="full_vec")
    # chunk-level embedding: union across chunks
    vec_rows = []
    for ch in chunks:
        vec_rows.append(recall_vector(embed(ch), k=40, sim_threshold=0.30))
    chunk_vec  = (
        pd.concat(vec_rows, ignore_index=True)
          .drop_duplicates("pid")
          .assign(source="chunk_vec")
    )

    # 3) Combine & dedupe
    pool = pd.concat([bm25_full, chunk_bm25, full_vec, chunk_vec], ignore_index=True)
    candidates = (
        pool
        .sort_values(["source","sim"], ascending=[True, False])
        .drop_duplicates("pid", keep="first")
        .reset_index(drop=True)
    )

    # 4) Metadata & filter
    meta   = fetch_metadata(candidates["pid"].tolist())
    merged = (
        candidates
        .merge(meta[["pid","abstract","year"]], on="pid", how="left")
        .dropna(subset=["abstract"])
    )
    if target_year is not None:
        merged = merged[merged["year"] <= target_year]
    if merged.empty:
        raise ValueError("No candidates after filtering")

    # 5) Contextual LLM rerank
    try:
        rer = llm_contextual_rerank(
            paragraph,
            merged[["pid","title","abstract"]],
            max_candidates=len(merged)
        )
        predicted = [pid for pid in rer["pid"] if pid in merged["pid"].tolist()]
    except Exception:
        predicted = merged["pid"].tolist()

    # 6) Prepare reference embeddings
    str_refs = [str(p) for p in true_pids]
    ref_meta = fetch_metadata(str_refs).dropna(subset=["abstract"])
    n_rel    = len(ref_meta)
    ref_embs = np.stack([embed(a) for a in ref_meta["abstract"]]) if n_rel else np.zeros((0,768))

    # 7) Compute sim_map & metrics
    cand_absts = merged.set_index("pid").loc[predicted, "abstract"].tolist()
    cand_embs  = np.stack([embed(a) for a in cand_absts])
    sims       = cosine_matrix(cand_embs, ref_embs) if n_rel else np.zeros((len(predicted),0))
    max_sims   = sims.max(axis=1) if n_rel else np.zeros(len(predicted))
    sim_map    = {pid: (s>=SIM_THRESHOLD) for pid,s in zip(predicted, max_sims)}

    results = {}
    for k in TOPK_LIST:
        results[f"P@{k}"]    = precision_by_sim(predicted, sim_map, k)
        results[f"R@{k}"]    = recall_by_sim(predicted, sim_map, k, n_rel)
        results[f"HR@{k}"]   = hit_rate_by_sim(predicted, sim_map, k)
        results[f"NDCG@{k}"] = ndcg_by_sim(predicted, sim_map, k)
    return results


def main():
    df = pd.read_json(TESTSET_PATH, lines=True).head(MAX_CASES)
    all_m = []
    for case in df.to_dict("records"):
        yr = case.get("year") or fetch_year_by_doi(case.get("doi"))
        all_m.append(evaluate_case(case["paragraph"], case.get("references", []), yr))

    avg = pd.DataFrame(all_m).mean(numeric_only=True)
    print("\nContextual LLM rerank average metrics over four recall sources:\n")
    print(avg)

if __name__ == "__main__":
    main()
