
# This is evaluation code

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from neo4j import GraphDatabase
from transformers import AutoTokenizer
from chunking import clean_text
from recall import (
    recall_fulltext,
    fetch_metadata,
    embed,
)
from rerank_llm import rerank_batch, RerankError  # returns DataFrame with pid, score
import matplotlib.pyplot as plt         


# config
TESTSET_PATH  = Path(os.getenv("TESTSET_PATH", "/home/abhi/Desktop/Manami/recommender-system/datasets/testset_2020_references.jsonl"))
MAX_CASES     = int(os.getenv("MAX_CASES", 25)) # Number of test cases to evaluate
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", 0.95))
TOPK_LIST     = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20) # K-values for evaluation metrics


# Neo4j connection
_NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
_NEO4J_PASS = os.getenv("NEO4J_PASS", "secret")
_driver     = GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASS))

# Tokenizer for chunking
TOKENIZER  = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", use_fast=True)

# helpers
def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T

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
        .merge(meta[["pid", "title", "abstract", "year"]], on="pid", how="left")
        .dropna(subset=["abstract"])
    )
    if target_year is not None:
        merged = merged[merged["year"] < target_year]
    if merged.empty:
        raise ValueError("Empty candidate set after filtering by year and abstract")

    try:
        reranked = rerank_batch(
            paragraph,
            merged[["pid","title","abstract"]],
            k=20,
            # batch_size=…  you can tune this
        )
        final_pids = reranked["pid"].tolist()
    except RerankError as e:
        print("⚠️ Rerank failed, fallback to BM25 order:", e)
        final_pids = merged["pid"].tolist()

    # 3) fetch and embed all the true reference papers’ abstracts. 
    # If none have valid abstracts, create a zero-matrix so that nothing is ever “similar.”
    ref_meta = fetch_metadata([str(p) for p in true_pids]).dropna(subset=["abstract"])
    ref_ids  = list(ref_meta["pid"])
    ref_embs = np.stack([embed(a) for a in ref_meta["abstract"]]) if ref_ids else np.zeros((0,768))


    # 7) Embed each candidate’s abstract
    cand_meta = merged.set_index("pid").loc[final_pids]
    cand_absts = cand_meta["abstract"].tolist()
    cand_embs  = np.stack([embed(a) for a in cand_absts])

    # 8) Compute full similarity matrix: candidates × references
    sims      = cosine_matrix(cand_embs, ref_embs) if ref_ids else np.zeros((len(final_pids),0))
    unmatched = set(range(len(ref_ids)))
    hits      = []

    for i in range(len(final_pids)):
        if not unmatched:
            hits.append(False)
            continue
        ref_idxs = list(unmatched)
        sim_vals = sims[i, ref_idxs]
        j = sim_vals.argmax()
        if sim_vals[j] >= SIM_THRESHOLD:
            hits.append(True)
            unmatched.remove(ref_idxs[j])
        else:
            hits.append(False)

    # https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems    
    # P@k  = # relevant hits in top-k / k
    # HR@k = # whether at least one hit in top-k
    # R@k  = # hits / total ground-truth
    # NDCG = # relevance-weighted discounted ranking quality
    n_rel = len(ref_ids)
    out   = {}
    for k in TOPK_LIST:
        topk = hits[:k]
        p_at_k = sum(topk) / k
        hr_at_k = float(any(topk))
        r_at_k = sum(topk) / n_rel if n_rel else 0.0
        dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(topk))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(n_rel, k)))
        ndcg = (dcg / idcg) if idcg else 0.0
        out.update({f"P@{k}": p_at_k, f"HR@{k}": hr_at_k, f"R@{k}": r_at_k, f"NDCG@{k}": ndcg})
    return out

# main function
# Runs evaluation over the test cases and prints the average metrics.
import matplotlib
matplotlib.use("Agg") 

def main() -> None:
    if not TESTSET_PATH.exists():
        raise FileNotFoundError(f"Testset not found at {TESTSET_PATH}")
    df = pd.read_json(TESTSET_PATH, lines=True).head(MAX_CASES)
    metrics = []
    for rec in df.to_dict("records"):
        metrics.append(
            evaluate_case(
                rec["paragraph"],
                [str(pid) for pid in rec.get("references", [])],
                rec.get("year")
            )
        )
    metric_df = pd.DataFrame(metrics)
    ks = np.array(TOPK_LIST)

    for prefix in ["P", "HR", "R", "NDCG"]:
        y = metric_df[[f"{prefix}@{k}" for k in ks]].mean().values

        # 1) create & plot
        plt.figure()
        plt.plot(ks, y, marker="o")
        plt.title(f"{prefix}@k vs k  (averaged over {len(metric_df)} paragraphs)")
        plt.xlabel("k: # of recommended papers")
        plt.ylabel(prefix)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 2) save and close
        plt.savefig(Path(__file__).parent / "eval" / f"{prefix.lower()}_atk.png", dpi=200)
        plt.close()

    avg = pd.DataFrame(metrics).mean(numeric_only=True)
    print("\nEvaluation with RRF + LLM scoring - avg metrics:\n", avg.round(4))


if __name__ == "__main__":
    main()
