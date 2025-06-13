# eval_hybrid.py (fixed)
# ---------------------------------------------------------------------------
# Evaluate hybrid BM25+embedding per-chunk (union)
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

TESTSET_PATH   = Path("/home/abhi/Desktop/Manami/recommender-system/datasets/testset_300_references.jsonl")
MAX_CASES      = 30
SIM_THRESHOLD  = 0.95
TOPK_LIST      = (3, 5, 10, 15, 20)

NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "Manami1008")
driver     = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# Initialize tokenizer once
from transformers import AutoTokenizer
TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# ─────────────────────────── METRICS ────────────────────────────────────── #

def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T

def precision_by_sim(pred, sim_map, k):
    return sum(sim_map.get(p, False) for p in pred[:k]) / k if k else 0.0

def recall_by_sim(pred, sim_map, k, n_rel):
    if n_rel == 0:
        return 0.0
    return sum(sim_map.get(p, False) for p in pred[:k]) / n_rel

def hit_rate_by_sim(pred, sim_map, k):
    return float(any(sim_map.get(p, False) for p in pred[:k]))

def ndcg_by_sim(pred, sim_map, k):
    rels  = [1 if sim_map.get(p, False) else 0 for p in pred[:k]]
    dcg   = sum(r / np.log2(i+2) for i, r in enumerate(rels))
    ideal = sorted(rels, reverse=True)
    idcg  = sum(r / np.log2(i+2) for i, r in enumerate(ideal))
    return dcg/idcg if idcg > 0 else 0.0

def fetch_year_by_doi(doi: Optional[str]) -> Optional[int]:
    # Safely fetch year, return None if doi missing or error
    if not doi:
        return None
    try:
        query = "MATCH (p:Paper {doi:$doi}) RETURN p.year AS year"
        with driver.session() as sess:
            rec = sess.run(query, doi=doi).single()
        return rec["year"] if rec and rec.get("year") is not None else None
    except Exception:
        return None

# ─────────────────────────── EVALUATION ─────────────────────────────────── #

def evaluate_case(paragraph: str, true_pids: List[str], target_year: Optional[int]):
    cleaned = clean_text(paragraph)
    # Pass tokenizer to chunk_tokens
    chunks  = chunk_tokens(cleaned, TOKENIZER, win=128, stride=64)

    hybrid = recall_by_chunks(chunks, k_bm25=40, k_vec=40, sim_th=0.30)

    meta   = fetch_metadata(hybrid["pid"].tolist())
    merged = (
        hybrid
        .merge(meta[["pid","abstract","year"]], on="pid", how="left")
        .dropna(subset=["abstract"] )
    )
    if target_year is not None:
        merged = merged[merged["year"] <= target_year]

    # prepare references
    ref_meta = fetch_metadata([str(p) for p in true_pids]).dropna(subset=["abstract"])
    n_rel    = len(ref_meta)
    ref_embs = np.stack([embed(a) for a in ref_meta["abstract"]]) if n_rel else np.zeros((0,768))

    # compute similarity map
    cand_texts = merged["abstract"].tolist()
    cand_embs  = np.stack([embed(a) for a in cand_texts])
    sims = cosine_matrix(cand_embs, ref_embs) if n_rel else np.zeros((len(cand_embs),0))
    max_sims = sims.max(axis=1) if n_rel else np.zeros(len(cand_embs))
    sim_map = {pid:(s >= SIM_THRESHOLD) for pid, s in zip(merged["pid"], max_sims)}

    # metrics
    results = {}
    pids = merged["pid"].tolist()
    for k in TOPK_LIST:
        results[f"P@{k}"]    = precision_by_sim(pids, sim_map, k)
        results[f"R@{k}"]    = recall_by_sim(pids, sim_map, k, n_rel)
        results[f"HR@{k}"]   = hit_rate_by_sim(pids, sim_map, k)
        results[f"NDCG@{k}"] = ndcg_by_sim(pids, sim_map, k)
    return results


def main():
    df = pd.read_json(TESTSET_PATH, lines=True).head(MAX_CASES)
    all_m = []
    for case in df.to_dict("records"):
        yr = case.get("year") or fetch_year_by_doi(case.get("doi"))
        all_m.append(evaluate_case(case["paragraph"], case.get("references", []), yr))

    avg = pd.DataFrame(all_m).mean(numeric_only=True)
    print("\nHybrid BM25+vector per-chunk average metrics:\n")
    print(avg)

if __name__ == "__main__":
    main()
