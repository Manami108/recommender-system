# BM25 full search 

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from neo4j import GraphDatabase

from chunking import clean_text
from recall import recall_fulltext, fetch_metadata, embed
import matplotlib.pyplot as plt         

# Hard-coded testset path and params
TESTSET_PATH   = Path("/home/abhi/Desktop/Manami/recommender-system/datasets/testset_2020_references.jsonl")
MAX_CASES      = 5
SIM_THRESHOLD  = 0.95
TOPK_LIST     = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20) # K-values for evaluation metrics

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


def evaluate_case(
    paragraph: str,
    true_pids: List[str],
    target_year: Optional[int] = None
) -> dict:

    # take the raw paragraph, clean it (lower-casing, removing punctuation/stopwords, etc.), 
    # so that BM25 and embeddings both work on normalized text.
    cleaned = clean_text(paragraph)

    # 1) This runs a full-text BM25 search over my paragraph document index, returning the top 40 candidate paper IDs along with their BM25 scores.
    bm25 = recall_fulltext(cleaned, k=20)

    # 2) I pull each candidate’s abstract and publication year from Neo4j, 
    # drop any papers missing an abstract, and filter out papers published after the paragraph’s target year.
    meta   = fetch_metadata(bm25["pid"].tolist())
    merged = (
        bm25
        .merge(meta[["pid","abstract","year"]], on="pid", how="left")
        .dropna(subset=["abstract"])
    )
    if target_year is not None:
        merged = merged[merged["year"] < target_year]

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
    n_rel = len(ref_ids)
    results = {}
    for k in TOPK_LIST:
        topk = hits[:k]
        p_at_k = sum(topk) / k
        hr_at_k = float(any(topk))
        r_at_k = sum(topk) / n_rel if n_rel else 0.0
        dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(topk))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(n_rel, k)))
        ndcg = (dcg / idcg) if idcg else 0.0
        results.update({f"P@{k}": p_at_k, f"HR@{k}": hr_at_k, f"R@{k}": r_at_k, f"NDCG@{k}": ndcg})

    return results

def main() -> None:
    df = pd.read_json(TESTSET_PATH, lines=True).head(MAX_CASES)
    all_metrics = []

    for rec in df.to_dict("records"):
        metrics = evaluate_case(
            rec["paragraph"],
            [str(x) for x in rec.get("references", [])],
            rec.get("year")
        )
        all_metrics.append(metrics)

    metric_df = pd.DataFrame(all_metrics)
    print("\nBM25 + LLM rerank (k=20) average metrics:\n")
    print(metric_df.mean(numeric_only=True).round(4))

    # plotting (optional)
    ks = np.array(TOPK_LIST)
    for prefix in ["P","HR","R","NDCG"]:
        y = metric_df[[f"{prefix}@{k}" for k in ks]].mean().values
        plt.figure()
        plt.plot(ks, y, marker="o")
        plt.title(f"{prefix}@k vs k")
        plt.xlabel("k")
        plt.ylabel(prefix)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(Path(__file__).parent / "eval" / f"{prefix.lower()}_bm25.png", dpi=200)
        plt.close()

    rows: List[dict] = []
    for rec in df.to_dict("records"):
        m = evaluate_case(
            rec["paragraph"],
            [str(x) for x in rec.get("references", [])],
            rec.get("year")
        )
        m["method"] = "bm25_full"                   
        rows.append(m)

    # 3) build DataFrame and write to CSV
    metric_df = pd.DataFrame(rows)
    out_path = Path(__file__).parent / "csv" / "metrics_bm25_full.csv"
    metric_df.to_csv(out_path, index=False)

    # 4) print average metrics
    print("\nBM25 full-text average metrics:\n",
          metric_df.mean(numeric_only=True).round(4))

if __name__ == "__main__":
    main()
