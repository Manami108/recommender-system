# BM25 full search 

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from neo4j import GraphDatabase

from chunking import clean_text
from recall import recall_fulltext, fetch_metadata, embed

# Hard-coded testset path and params
TESTSET_PATH   = Path("/home/abhi/Desktop/Manami/recommender-system/datasets/testset_300_references.jsonl")
MAX_CASES      = 3
SIM_THRESHOLD  = 0.95
TOPK_LIST      = (3, 5, 10, 15, 20)

# Neo4j
NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "Manami1008")
driver     = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


# cosign similarity, precision, HR, recall, NDCG, year sort

def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T

def precision_by_sim(pred, sim_map, k):
    return sum(sim_map.get(p, False) for p in pred[:k]) / k if k else 0.0

def recall_by_sim(pred, sim_map, k, n_rel):
    if n_rel == 0: return 0.0
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
    if not doi: return None
    query = "MATCH (p:Paper {doi: $doi}) RETURN p.year AS year"
    with driver.session() as sess:
        rec = sess.run(query, doi=doi).single()
    return rec["year"] if rec and rec["year"] is not None else None


def evaluate_case(paragraph: str, true_pids: List[str], target_year: Optional[int]):
    # take the raw paragraph, clean it (lower-casing, removing punctuation/stopwords, etc.), 
    # so that BM25 and embeddings both work on normalized text.
    cleaned = clean_text(paragraph)

    # 1) This runs a full-text BM25 search over my paragraph document index, returning the top 40 candidate paper IDs along with their BM25 scores.
    bm25 = recall_fulltext(cleaned, k=40)

    # 2) I pull each candidate’s abstract and publication year from Neo4j, 
    # drop any papers missing an abstract, and filter out papers published after the paragraph’s target year.
    meta   = fetch_metadata(bm25["pid"].tolist())
    merged = (
        bm25
        .merge(meta[["pid","abstract","year"]], on="pid", how="left")
        .dropna(subset=["abstract"])
    )
    if target_year is not None:
        merged = merged[merged["year"] <= target_year]

    # 3) fetch and embed all the true reference papers’ abstracts. 
    # If none have valid abstracts, create a zero-matrix so that nothing is ever “similar.”
    ref_meta = fetch_metadata([str(p) for p in true_pids]).dropna(subset=["abstract"])
    n_rel    = len(ref_meta)
    ref_embs = np.stack([embed(a) for a in ref_meta["abstract"]]) if n_rel else np.zeros((0,768))

    # 4) embed each BM25 candidate’s abstract, compute a cosine similarity matrix against the true-ref embeddings, 
    # take each candidate’s maximum similarity to any reference, and mark it “relevant” if that max ≥ SIM_THRESHOLD (0.95).
    cand_absts = merged["abstract"].tolist()
    cand_embs  = np.stack([embed(a) for a in cand_absts])
    sims       = cosine_matrix(cand_embs, ref_embs) if n_rel else np.zeros((len(cand_embs),0))
    max_sims   = sims.max(axis=1) if n_rel else np.zeros(len(cand_absts))
    sim_map    = {pid: (s >= SIM_THRESHOLD) for pid, s in zip(merged["pid"], max_sims)}

    # 5) With binary relevance map, compute Precision@k, Recall@k (fraction of all true refs recovered), Hit-Rate@k (at least one “hit” in top-k), and NDCG@k.
    results = {}
    pids    = merged["pid"].tolist()
    for k in TOPK_LIST:
        results[f"P@{k}"]    = precision_by_sim(pids, sim_map, k)
        results[f"R@{k}"]    = recall_by_sim(pids, sim_map, k, n_rel)
        results[f"HR@{k}"]   = hit_rate_by_sim(pids, sim_map, k)
        results[f"NDCG@{k}"] = ndcg_by_sim(pids, sim_map, k)
    return results

def main():
    df = pd.read_json(TESTSET_PATH, lines=True).head(MAX_CASES)
    all_metrics = []
    for case in df.to_dict("records"):
        yr = case.get("year") or fetch_year_by_doi(case.get("doi"))
        all_metrics.append(evaluate_case(case["paragraph"], case.get("references",[]), yr))

    avg = pd.DataFrame(all_metrics).mean(numeric_only=True)
    print("\nBM25 full-text (k=40) average metrics:\n")
    print(avg)

if __name__ == "__main__":
    main()
