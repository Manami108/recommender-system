
# This is evaluation code

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from neo4j import GraphDatabase, READ_ACCESS
from transformers import AutoTokenizer
from chunking import clean_text, chunk_tokens
from recall import (
    recall_fulltext,
    recall_vector,
    recall_by_chunks,
    rrf_fuse, 
    fetch_metadata,
    embed,
)
from rerank_llm import rerank_batch, RerankError  # returns DataFrame with pid, score
import matplotlib.pyplot as plt         


# config
TESTSET_PATH  = Path(os.getenv("TESTSET_PATH", "/home/abhi/Desktop/Manami/recommender-system/datasets/testset_2020_references.jsonl"))
MAX_CASES     = int(os.getenv("MAX_CASES", 5)) # Number of test cases to evaluate
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
    # 1. clean & chunk paragraph
    cleaned = clean_text(paragraph)
    chunks  = chunk_tokens(cleaned, TOKENIZER, win=128, stride=64)

    # 2. recall & fuse with reciprocal rank fusion (RRF)
    # full-text & vector for whole paragraph
    full_bm25 = recall_fulltext(cleaned).assign(src="full_bm25")
    full_vec  = recall_vector(embed(cleaned)).assign(src="full_vec")
    
    # chunk-level retrieval
    chunk_pool = recall_by_chunks(chunks)

    # apply RRF fusion across all sources
    # full-doc scores (bm25_score, semantic_score) → hybrid_full
    # chunk-level max scores → hybrid_chunk
    # Total hybrid_score = hybrid_full + hybrid_chunk
    # strip out any extraneous columns so rrf_fuse sees just pid & rank
    # strip everything except pid & rank on the two full-doc tables
    for df in (full_bm25, full_vec):
        df.drop(columns=[c for c in df.columns if c not in ("pid","rank")],
                inplace=True)

    # now chunk_pool still has pid, rank, source, bm25_score, semantic_score
    # next drop source on the full-docs only:
    clean_full_bm25 = full_bm25.drop(columns=["source"], errors="ignore")
    clean_full_vec  = full_vec .drop(columns=["source"], errors="ignore")

    pool = rrf_fuse(
        clean_full_bm25,
        clean_full_vec,
        chunk_pool,      # still contains .source + .rank
        k_rrf=60,
        top_k=60
    )

    # keep top 60 candidates
    top_cands = pool["pid"].head(60).tolist()

    # Retrieves title, abstract, etc. of the 60 candidates.
    # Removes papers without abstracts.
    meta = fetch_metadata(top_cands)
    cand = meta.dropna(subset=["abstract"])
    if target_year is not None:
        cand = cand[cand["year"] < target_year]
    if cand.empty:
        raise ValueError("Empty candidate set after filtering by year and abstract")

    # 4. rerank via LLM scoring
    try:
        reranked = rerank_batch(
            paragraph,
            cand[["pid","title","abstract"]],
            k=60,
            max_candidates=60,
            # batch_size=20
        )
        predicted = reranked["pid"].tolist()
    except RerankError as e:
        print("⚠️ Rerank failed, fallback to pool order:", e)
        predicted = cand["pid"].tolist()

    # embed references & predicted candidates
    # Calculates cosine similarities between reranked abstracts and true references.
    # For each predicted paper, checks if any unmatched reference has sim ≥ 0.95 → it's a hit.
    ref_meta = fetch_metadata(true_pids)
    ref_meta = ref_meta.dropna(subset=["abstract"])
    ref_ids  = ref_meta["pid"].tolist()
    ref_emb  = (
        np.stack([embed(txt) for txt in ref_meta["abstract"]])
        if ref_ids else np.zeros((0, 768))
    )
    cand_meta = cand.set_index("pid").loc[predicted]
    cand_emb  = np.stack([embed(txt) for txt in cand_meta["abstract"].tolist()])

    # 6. greedy match for hit detection
    sims = cosine_matrix(cand_emb, ref_emb) if ref_ids else np.zeros((len(predicted),0))
    unmatched = set(range(len(ref_ids)))
    hits = []
    for i in range(len(predicted)):
        if not unmatched:
            hits.append(False)
            continue
        idxs = list(unmatched)
        best_j = sims[i, idxs].argmax()
        if sims[i, idxs][best_j] >= SIM_THRESHOLD:
            hits.append(True)
            unmatched.remove(idxs[best_j])
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




def evaluate_case(
    paragraph: str,
    true_pids: List[str],
    target_year: Optional[int] = None
) -> dict:
    # 0) normalize the paragraph
    cleaned = clean_text(paragraph)

    # 1) full‐text BM25 → top 20
    bm25 = recall_fulltext(cleaned, k=20)

    # 2) fetch metadata & filter
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

    # 3) LLM RERANK these 20 → final 20 in new order
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

    # 4) Prepare ground‐truth embeddings
    ref_meta = fetch_metadata([str(p) for p in true_pids]).dropna(subset=["abstract"])
    ref_ids  = ref_meta["pid"].tolist()
    ref_embs = (
        np.stack([embed(a) for a in ref_meta["abstract"]])
        if ref_ids else np.zeros((0,768))
    )

    # 5) Candidate abstracts & embeddings in reranked order
    cand_meta = merged.set_index("pid").loc[final_pids]
    cand_absts = cand_meta["abstract"].tolist()
    cand_embs  = np.stack([embed(a) for a in cand_absts])

    # 6) Greedy one‐to‐one matching
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

    # 7) Metrics
    n_rel = len(ref_ids)
    out   = {}
    for k in TOPK_LIST:
        topk    = hits[:k]
        p_at_k  = sum(topk)/k
        hr_at_k = float(any(topk))
        r_at_k  = sum(topk)/n_rel if n_rel else 0.0
        dcg     = sum(rel/np.log2(idx+2) for idx,rel in enumerate(topk))
        idcg    = sum(1/np.log2(i+2) for i in range(min(n_rel,k)))
        ndcg    = (dcg/idcg) if idcg else 0.0
        out[f"P@{k}"]    = p_at_k
        out[f"HR@{k}"]   = hr_at_k
        out[f"R@{k}"]    = r_at_k
        out[f"NDCG@{k}"] = ndcg

    return out
