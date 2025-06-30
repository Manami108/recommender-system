
# This is full BM25 only - no hops, no chunks but llm reranking 
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
MAX_CASES     = int(os.getenv("MAX_CASES", 100)) # Number of test cases to evaluate
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", 0.95))
TOPK_LIST     = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20) # K-values for evaluation metrics

# Neo4j connection
_NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
_NEO4J_PASS = os.getenv("NEO4J_PASS", "secret")
_driver     = GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASS))

# Tokenizer for chunking
TOKENIZER  = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", use_fast=True)

# cosine similarity 
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

    # This runs a full-text BM25 search over my paragraph document index, returning the top 40 candidate paper IDs (because many abstract might be missed) along with their BM25 scores.
    # BM25 does consider paragraph length as well so better than tf-idf
    bm25 = recall_fulltext(cleaned, k=50)

    # I pull each candidate’s abstract and publication year from Neo4j, 
    # drop any papers missing an abstract. 
    meta   = fetch_metadata(bm25["pid"].tolist())
    merged = (
        bm25
        .merge(meta[["pid", "title", "abstract", "year"]], on="pid", how="left")
        .dropna(subset=["abstract"])
        .sort_values("bm25_score", ascending=False)   # explicit sort
        .head(20)
    )

    # This fillter out the future papers but now the year is set to 2020 (latest in the dataset) so it does not matter. 
    if target_year is not None:
        merged = merged[merged["year"] < target_year]
    if merged.empty:
        raise ValueError("Empty candidate set after filtering by year and abstract")
    
    # Ask the LLM to rerank your 20 candidates by relevance to the paragraph.
    try:
        llm_df = rerank_batch(
            paragraph,
            merged[["pid", "title", "abstract"]],
            k=20,                   # let the LLM see / rank all 20
        )                           # → columns: pid, score

        # add BM25 score for deterministic tie-breaking
        llm_df = llm_df.merge(
            merged[["pid", "bm25_score"]],
            on="pid",
            how="left",
        )

        # sort: 1) LLM score ↓  2) bm25_score ↓
        llm_df = llm_df.sort_values(
            ["score", "bm25_score"],
            ascending=[False, False],
            kind="mergesort",
        )

        predicted = llm_df["pid"].tolist()

    except RerankError as e:
        print("⚠️  Rerank failed, falling back to BM25 order:", e)
        predicted = merged["pid"].tolist()

    # fetch and embed all the true reference papers’ abstracts. 
    # If none have valid abstracts, create a zero-matrix so that nothing is ever “similar.”
    ref_meta = fetch_metadata([str(p) for p in true_pids]).dropna(subset=["abstract"])
    ref_ids  = ref_meta["pid"].tolist()
    ref_embs = np.stack([embed(a) for a in ref_meta["abstract"]]) if ref_ids else np.zeros((0, 768))

    cand_meta  = merged.set_index("pid").loc[predicted]
    cand_embs  = np.stack([embed(a) for a in cand_meta["abstract"]])

    # Keep the references still unmatched 
    # If the similarity is more than 0.95, mark that candidate as a hit and remove the reference from future matching.

    sims = cosine_matrix(cand_embs, ref_embs) if ref_ids else np.zeros((len(predicted), 0))
    unmatched, hits = set(range(len(ref_ids))), []
    for i in range(len(predicted)):
        if not unmatched:
            hits.append(False)
            continue
        idxs = list(unmatched)
        j = sims[i, idxs].argmax()
        if sims[i, idxs][j] >= SIM_THRESHOLD:
            hits.append(True)
            unmatched.remove(idxs[j])
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


import matplotlib
matplotlib.use("Agg") 

# main function
# Runs evaluation over the test cases and prints the average metrics.
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

    rows: List[dict] = []
    for rec in df.to_dict("records"):
        m = evaluate_case(
            rec["paragraph"],
            [str(pid) for pid in rec.get("references", [])],
            rec.get("year")
        )
        m["method"] = "bm25_llm"  
        rows.append(m)

    for prefix in ["P", "HR", "R", "NDCG"]:
        y = metric_df[[f"{prefix}@{k}" for k in ks]].mean().values

        # create & plot
        plt.figure()
        plt.plot(ks, y, marker="o")
        plt.title(f"{prefix}@k vs k  (averaged over {len(metric_df)} paragraphs)")
        plt.xlabel("k: # of recommended papers")
        plt.ylabel(prefix)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # save and close
        plt.savefig(Path(__file__).parent / "eval" / f"{prefix.lower()}_bm25_llm.png", dpi=200)
        plt.close()

    avg = pd.DataFrame(metrics).mean(numeric_only=True)
    print("\nEvaluation with BM25 + LLM scoring - avg metrics:\n", avg.round(4))

    # build DataFrame and write to CSV
    metric_df = pd.DataFrame(rows)
    out_csv   = Path(__file__).parent / "csv" / "metrics_bm25_llm.csv"
    metric_df.to_csv(out_csv, index=False)
    print(f"\nSaved per‐paragraph metrics to {out_csv}")

    avg = metric_df.mean(numeric_only=True).round(4)
    print("\nAverage metrics:\n", avg)

if __name__ == "__main__":
    main()
