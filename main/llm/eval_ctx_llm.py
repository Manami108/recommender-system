#!/usr/bin/env python
# eval_recommender.py
# ---------------------------------------------------------------------------
# Offline evaluation for context-aware paper recommender
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

# ─────────────────────────── CONFIGURATION ──────────────────────────── #

NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "Manami1008")

SIM_THRESHOLD = 0.95          # cosine sim threshold for “correct”
TOPK_LIST     = (3, 5, 10, 15, 20)   # metrics to report
TOKENIZER     = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


default_path = Path(os.getenv("TESTSET_PATH",
                  "/home/abhi/Desktop/Manami/recommender-system/datasets/testset_300_references.jsonl"))

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
    debug: bool = False,
) -> dict:
    # 1) Pre-process & chunk
    cleaned = clean_text(paragraph)
    chunks  = chunk_tokens(cleaned, TOKENIZER, win=128, stride=64)

    # 2) Recall candidates
    bm25_full = recall_fulltext(cleaned, k=40).assign(source="bm25_full")
    vec_full  = recall_vector(embed(cleaned), k=40, sim_threshold=0.30).assign(source="embed_full")
    chunked   = recall_by_chunks(chunks, k_bm25=40, k_vec=40, sim_th=0.30).assign(source="chunked")

    candidates = (
        pd.concat([bm25_full, vec_full, chunked], ignore_index=True)
          .sort_values(["source","sim"], ascending=[True, False])
          .drop_duplicates("pid",     keep="first")
          .reset_index(drop=True)
    )

    # 3) Fetch metadata & drop missing abstracts
    meta   = fetch_metadata(candidates["pid"].tolist())
    merged = (
        candidates
          .merge(meta[["pid","abstract","year"]], on="pid", how="left")
          .dropna(subset=["abstract"])
    )

    # 4) Year filter
    if target_year is not None:
        merged = merged[merged["year"] <= target_year]
    if merged.empty:
        raise ValueError("No valid candidates after year/abstract filtering")

    # 5) LLM rerank (falls back to original order on error)
    try:
        reranked = llm_contextual_rerank(
            paragraph,
            merged[["pid","title","abstract"]],
            max_candidates=40,
        )
        predicted = [pid for pid in reranked["pid"] if pid in merged["pid"].tolist()]
    except Exception:
        predicted = merged["pid"].tolist()

    # 6) Reference embeddings (skip refs w/o abstracts)
    #    Ensure IDs are strings so fetch_metadata matches your CSV-imported p.id
    str_refs = [str(pid) for pid in true_pids]
    ref_meta = fetch_metadata(str_refs).dropna(subset=["abstract"])
    if ref_meta.empty:
        ref_embs = np.zeros((0, 768), dtype=np.float32)
    else:
        ref_embs = np.stack([ embed(ab) for ab in ref_meta["abstract"] ])

    # 7) Candidate vs. ref similarity (vectorised)
    cand_abst = merged.set_index("pid").loc[predicted, "abstract"]
    cand_embs = np.stack([ embed(ab) for ab in cand_abst ])

    if ref_embs.size == 0:
        max_sims = np.zeros(len(predicted), dtype=float)
    else:
        sims     = cosine_matrix(cand_embs, ref_embs)  # (N_rec, N_ref)
        max_sims = sims.max(axis=1)

    sim_map = {pid: (sim >= SIM_THRESHOLD) for pid, sim in zip(predicted, max_sims)}

    # 8) Compute metrics
    results: dict[str, float] = {}
    n_rel = len([pid for pid in true_pids if pid in sim_map and sim_map[pid]])
    for k in TOPK_LIST:
        results[f"P@{k}"]     = precision_by_sim(predicted, sim_map, k)
        results[f"R@{k}"]     = recall_by_sim(predicted, sim_map, k, n_rel)
        results[f"HR@{k}"]    = hit_rate_by_sim(predicted, sim_map, k)
        results[f"NDCG@{k}"]  = ndcg_by_sim(predicted, sim_map, k)

    # Attach debug tables if requested
    if debug:
        results["_candidates"] = candidates
        results["_merged"]     = merged
        results["_ref_meta"]   = ref_meta

    return results


# ────────────────────────────────── MAIN ────────────────────────────────── #

def main(testset_path: Path, max_cases: int, debug: bool):
    df = pd.read_json(testset_path, lines=True).head(max_cases)

    all_metrics = []
    for case in df.to_dict(orient="records"):
        para = case["paragraph"]
        refs = case.get("references", [])
        cid  = case.get("id") or case.get("doi")
        # If your testset rows have a doi field, fetch that year; else None
        year = case.get("year") or fetch_year_by_doi(case.get("doi"))

        metrics = evaluate_case(para, refs, target_year=year, debug=debug)
        if debug:
            trace_case(
                case_id=cid,
                true_pids=set(str(r) for r in refs),
                candidates=metrics.pop("_candidates"),
                merged=metrics.pop("_merged"),
                ref_meta=metrics.pop("_ref_meta"),
                target_year=year,
            )

        metrics["case_id"] = cid
        # Remove sim_map so mean() stays numeric-only
        metrics.pop("sim_map", None)
        all_metrics.append(metrics)

    res_df = pd.DataFrame(all_metrics).set_index("case_id")
    print("\nPer-case metrics:")
    print(res_df)

    print("\nAverage metrics:")
    print(res_df.mean(numeric_only=True))


# ───────────────────────────────── CLI ───────────────────────────────── #

if __name__ == "__main__":
    # 1) pick up TESTSET_PATH env or fallback to hard-coded path
    testset_path = default_path

    # 2) set your defaults here
    max_cases = 3
    debug     = False

    if not testset_path.is_file():
        raise FileNotFoundError(f"Testset not found: {testset_path!r}")

    main(testset_path, max_cases=max_cases, debug=debug)