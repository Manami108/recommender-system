# BM25 full search 

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from neo4j import GraphDatabase

from chunking import clean_text
from recall import recall_fulltext, recall_vector, recall_by_chunks, hybrid_rank, fetch_metadata, embed, rrf_fuse


# Hard-coded testset path and params
TESTSET_PATH   = Path("/home/abhi/Desktop/Manami/recommender-system/datasets/testset_2020_references.jsonl")
MAX_CASES      = 25
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


def evaluate_case(paragraph: str, true_pids: List[str]):
    # take the raw paragraph, clean it (lower-casing, removing punctuation/stopwords, etc.), 
    # so that BM25 and embeddings both work on normalized text.
    cleaned = clean_text(paragraph)

    # 1) Retrieve BM25 and vector hits for the *whole* paragraph
    bm25 = recall_fulltext(cleaned, k=20)
    vec  = recall_vector(embed(cleaned), k=20)

    chunks     = [cleaned]  # or use your chunk_tokens() if you have it
    chunk_pool = recall_by_chunks(chunks, k_bm25=10, k_vec=10, sim_th=SIM_THRESHOLD)

    # 2) Fuse via Reciprocal Rank Fusion (RRF)
    rrf = rrf_fuse(
        bm25,
        vec,
        chunk_pool,
        k_rrf=60,
        top_k=20,
    )

    # 3) Take top-20 by the RRF score
    top_pids = rrf["pid"].tolist()

    # 4) Fetch abstracts for these top candidates
    merged = (
        fetch_metadata(top_pids)
        .loc[lambda df: df.abstract.ne("")]  # drop any missing abstracts
    )


    # 3) fetch and embed all the true reference papers’ abstracts. 
    # If none have valid abstracts, create a zero-matrix so that nothing is ever “similar.”
    ref_meta = fetch_metadata([str(p) for p in true_pids]).dropna(subset=["abstract"])
    ref_ids  = list(ref_meta["pid"])
    ref_embs = np.stack([embed(a) for a in ref_meta["abstract"]]) if ref_ids else np.zeros((0,768))


    # 7) Embed each candidate’s abstract
    cand_ids   = merged["pid"].tolist()
    cand_absts = merged["abstract"].tolist()
    cand_embs  = np.stack([embed(a) for a in cand_absts])

    # 8) Compute full similarity matrix: candidates × references
    sims = cosine_matrix(cand_embs, ref_embs)  # shape (n_cand, n_ref)

    # 9) Greedy one‐to‐one matching at threshold
    unmatched = set(range(len(ref_ids)))   # indices of refs not yet covered
    hits      = []  # for each candidate, store whether it “covers” a new ref

    for i, pid in enumerate(cand_ids):
        if not unmatched:
            hits.append(False)
            continue

        # look only at still‐unmatched references
        ref_idxs = list(unmatched)
        sim_vals = sims[i, ref_idxs]
        j = sim_vals.argmax()  # index within ref_idxs
        best_sim = sim_vals[j]
        if best_sim >= SIM_THRESHOLD:
            # mark both candidate as hit, and remove that ref
            hits.append(True)
            unmatched.remove(ref_idxs[j])
        else:
            hits.append(False)

    # 10) Build metrics using this “hits” list
    results = {}
    for k in TOPK_LIST:
        topk = hits[:k]
        n_rel = len(ref_ids)
        results[f"P@{k}"]  = sum(topk)/k
        # how many distinct refs we covered in top‐k?
        covered = min(len(ref_ids), sum(topk))
        results[f"R@{k}"]  = covered / n_rel if n_rel else 0.0
        results[f"HR@{k}"] = float(any(topk))
        # For NDCG, build relevance array r_i ∈ {0,1}
        dcg  = sum(r/np.log2(i+2) for i,r in enumerate(topk))
        ideal = sorted(topk, reverse=True)
        idcg = sum(r/np.log2(i+2) for i,r in enumerate(ideal))
        results[f"NDCG@{k}"] = dcg/idcg if idcg>0 else 0.0

    return results

def main():
    df = pd.read_json(TESTSET_PATH, lines=True).head(MAX_CASES)
    all_metrics = []
    for case in df.to_dict("records"):
        all_metrics.append(evaluate_case(case["paragraph"], case.get("references",[])))

    avg = pd.DataFrame(all_metrics).mean(numeric_only=True)
    print("\nBM25 full-text (k=20) average metrics:\n")
    print(avg)

if __name__ == "__main__":
    main()
