#!/usr/bin/env python
# eval_recommender_compare.py
# ---------------------------------------------------------------------------
# Offline evaluation for context-aware paper recommender
# Compares two LLM-based rerankers:
#  1) contextual prompt + JSON parsing
#  2) log-likelihood scoring
# ---------------------------------------------------------------------------

import argparse
import os
from pathlib import Path
from typing import List, Set, Optional

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from neo4j import GraphDatabase

from chunking import clean_text, chunk_tokens
from recall import (
    recall_by_chunks,
    recall_fulltext,
    recall_vector,
    embed,
    fetch_metadata,
)
from rerank_llm import llm_contextual_rerank

# Import the new log-likelihood reranker
from llm_normal import llm_loglikelihood_rerank  


# ─────────────────────────── CONFIGURATION ──────────────────────────── #

NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "Manami1008")

SIM_THRESHOLD = 0.95
TOPK_LIST     = (3, 5, 10, 15, 20)
TOKENIZER     = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

default_path = Path(os.getenv(
    "TESTSET_PATH",
    "/home/abhi/Desktop/Manami/recommender-system/datasets/testset_300_references.jsonl"
))
parser = argparse.ArgumentParser()
parser.add_argument(
    "--testset",
    type=Path,
    default=default_path,
    help="JSONL file with paragraph + references (can also set TESTSET_PATH env var)",
)
# ──────────────────────────── SMALL HELPERS ─────────────────────────── #

def fetch_year_by_doi(doi: Optional[str]) -> Optional[int]:
    """
    If your testset uses DOIs (and you exported doi on Paper nodes),
    this returns the year for that DOI.  Otherwise, returns None.
    """
    if not doi:
        return None

    query = """
    MATCH (p:Paper {doi: $doi})
    RETURN p.year AS year
    """
    with driver.session() as sess:
        rec = sess.run(query, doi=doi).single()
    return rec["year"] if rec and rec["year"] is not None else None


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Cosine similarity matrix for two L2-normalised arrays.
    a: (N, d), b: (M, d)  →  result: (N, M)
    """
    return a @ b.T


# ─────────────────────────── METRIC UTILITIES ─────────────────────────── #

def precision_by_sim(pred: List[str], sim_map: dict[str, bool], k: int) -> float:
    flags = [sim_map.get(pid, False) for pid in pred[:k]]
    return sum(flags) / k if k else 0.0


def hit_rate_by_sim(pred: List[str], sim_map: dict[str, bool], k: int) -> float:
    return float(any(sim_map.get(pid, False) for pid in pred[:k]))

def recall_by_sim(pred: List[str], sim_map: dict[str,bool], k: int, n_rel: int) -> float:
    """Recall@k = # of relevant in top-k ÷ total relevant."""
    if n_rel == 0:
        return 0.0
    hits = sum(1 for pid in pred[:k] if sim_map.get(pid, False))
    return hits / n_rel
import math

def ndcg_by_sim(pred: List[str], sim_map: dict[str,bool], k: int) -> float:
    """
    NDCG@k: discounted gain of relevance judgements (binary) 
    normalized by ideal DCG@k.
    """
    # relevance of each position 1..k
    rels = [1 if sim_map.get(pid, False) else 0 for pid in pred[:k]]
    # DCG = sum rel_i / log2(i+1), with i starting from 1
    dcg = sum(rel / math.log2(idx+2) for idx, rel in enumerate(rels))
    # IDCG: best possible ordering = all ones up to min(k, total rels)
    ideal_rels = sorted(rels, reverse=True)
    idcg = sum(rel / math.log2(idx+2) for idx, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0

# ──────────────────────────── DEBUG TRACING ─────────────────────────── #

def trace_case(
    case_id: str,
    true_pids: Set[str],
    candidates: pd.DataFrame,
    merged: pd.DataFrame,
    ref_meta: pd.DataFrame,
    target_year: Optional[int],
):
    print(f"\n── DEBUG trace for case {case_id} ─────────────────────────")
    print(f"Ground-truth refs        : {len(true_pids)}")
    print(f"Hits after recall        : {len(true_pids & set(candidates['pid']))}")
    print(f"Hits after year/abs filter: {len(true_pids & set(merged['pid']))}")
    print(f"Target year              : {target_year}")
    print(f"Refs lacking abstracts   : {ref_meta['abstract'].isna().sum()}")
    print("────────────────────────────────────────────────────────────")


# ───────────────────────────── EVALUATION ───────────────────────────── #

def evaluate_case(
    paragraph: str,
    true_pids: List[str],
    target_year: Optional[int] = None,
) -> dict:
    """
    Returns a dict containing metrics for both rerankers:
      - keys prefixed with 'ctx_' for llm_contextual_rerank
      - keys prefixed with 'll_' for llm_loglikelihood_rerank
    """
    # 1) preprocess & chunk
    cleaned = clean_text(paragraph)
    chunks  = chunk_tokens(cleaned, TOKENIZER, win=128, stride=64)

    # 2) recall candidates
    bm25_full = recall_fulltext(cleaned, k=40).assign(source="bm25_full")
    vec_full  = recall_vector(embed(cleaned), k=40, sim_threshold=0.30).assign(source="embed_full")
    chunked   = recall_by_chunks(chunks, k_bm25=40, k_vec=40, sim_th=0.30).assign(source="chunked")

    candidates = (
        pd.concat([bm25_full, vec_full, chunked], ignore_index=True)
          .sort_values(["source","sim"], ascending=[True, False])
          .drop_duplicates("pid", keep="first")
          .reset_index(drop=True)
    )

    # 3) fetch metadata & filter
    meta   = fetch_metadata(candidates["pid"].tolist())
    merged = (
        candidates
          .merge(meta[["pid","abstract","year"]], on="pid", how="left")
          .dropna(subset=["abstract"])
    )
    if target_year is not None:
        merged = merged[merged["year"] <= target_year]
    if merged.empty:
        raise ValueError("No valid candidates after filtering")

    # prepare true reference set and embeddings
    str_refs = [str(pid) for pid in true_pids]
    ref_meta = fetch_metadata(str_refs).dropna(subset=["abstract"])
    n_rel = len(ref_meta)
    if n_rel > 0:
        ref_embs = np.stack([embed(ab) for ab in ref_meta["abstract"]])
    else:
        ref_embs = np.zeros((0, 768), dtype=float)

    results: dict[str, float] = {}

    # helper to compute sim_map and metrics given a ranked list of pids
    def compute_metrics(predicted: List[str], prefix: str):
        # compute candidate- vs ref-similarity
        if ref_embs.size:
            cand_abst = merged.set_index("pid").loc[predicted, "abstract"]
            cand_embs = np.stack([embed(ab) for ab in cand_abst])
            sims = cosine_matrix(cand_embs, ref_embs)
            max_sims = sims.max(axis=1)
        else:
            max_sims = np.zeros(len(predicted), dtype=float)

        sim_map = {pid: (sim >= SIM_THRESHOLD) for pid, sim in zip(predicted, max_sims)}

        for k in TOPK_LIST:
            results[f"{prefix}P@{k}"]    = precision_by_sim(predicted, sim_map, k)
            results[f"{prefix}R@{k}"]    = recall_by_sim(predicted, sim_map, k, n_rel)
            results[f"{prefix}HR@{k}"]   = hit_rate_by_sim(predicted, sim_map, k)
            results[f"{prefix}NDCG@{k}"] = ndcg_by_sim(predicted, sim_map, k)

    # 5a) contextual-prompt rerank
    try:
        rer_ctx = llm_contextual_rerank(
            paragraph,
            merged[["pid","title","abstract"]],
            max_candidates=40,
        )["pid"].tolist()
    except Exception:
        rer_ctx = merged["pid"].tolist()
    compute_metrics(rer_ctx, prefix="ctx_")

    # 5b) log-likelihood rerank
    try:
        rer_ll = llm_loglikelihood_rerank(
            paragraph,
            merged[["pid","title","abstract"]],
            max_candidates=40,
        )["pid"].tolist()
    except Exception:
        rer_ll = merged["pid"].tolist()
    compute_metrics(rer_ll, prefix="ll_")

    return results


# ────────────────────────────────── MAIN ────────────────────────────────── #

def main(testset_path: Path, max_cases: int):
    df = pd.read_json(testset_path, lines=True).head(max_cases)
    all_metrics = []

    for case in df.to_dict(orient="records"):
        para = case["paragraph"]
        refs = case.get("references", [])
        year = case.get("year") or fetch_year_by_doi(case.get("doi"))
        metrics = evaluate_case(para, refs, target_year=year)
        metrics["case_id"] = case.get("id") or case.get("doi")
        all_metrics.append(metrics)

    res_df = pd.DataFrame(all_metrics).set_index("case_id")

    print("\nAverage metrics (contextual-prompt vs. log-likelihood):")
    # select only P@, R@, etc
    avg = res_df.mean(numeric_only=True)
    # separate into two DataFrames for clarity
    ctx_avg = avg.filter(like="ctx_").rename(lambda k: k.replace("ctx_", ""))
    ll_avg  = avg.filter(like="ll_").rename(lambda k: k.replace("ll_", ""))
    comp = pd.DataFrame({"contextual": ctx_avg, "loglikelihood": ll_avg})
    print(comp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", type=Path, default=default_path)
    parser.add_argument("--max_cases", type=int, default=40)
    args = parser.parse_args()

    if not args.testset.is_file():
        raise FileNotFoundError(f"Testset not found: {args.testset!r}")
    main(args.testset, args.max_cases)