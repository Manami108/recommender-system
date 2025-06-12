#!/usr/bin/env python
# eval_recommender_compare.py
# ---------------------------------------------------------------------------
# Offline evaluation for context-aware paper recommender
# Compares four ranking strategies:
#  1) pure BM25
#  2) pure embedding vector similarity
#  3) LLM contextual prompt + JSON parsing
#  4) LLM log-likelihood scoring
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
from llm_normal import llm_loglikelihood_rerank  # your log-likelihood reranker

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


def fetch_year_by_doi(doi: Optional[str]) -> Optional[int]:
    if not doi:
        return None
    query = "MATCH (p:Paper {doi: $doi}) RETURN p.year AS year"
    with driver.session() as sess:
        rec = sess.run(query, doi=doi).single()
    return rec["year"] if rec and rec["year"] is not None else None


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T


def precision_by_sim(pred, sim_map, k): return sum(sim_map.get(p, False) for p in pred[:k]) / k if k else 0.0
def hit_rate_by_sim(pred, sim_map, k):    return float(any(sim_map.get(p, False) for p in pred[:k]))
def recall_by_sim(pred, sim_map, k, n_rel):
    if n_rel == 0: return 0.0
    return sum(sim_map.get(p, False) for p in pred[:k]) / n_rel
import math
def ndcg_by_sim(pred, sim_map, k):
    rels = [1 if sim_map.get(p, False) else 0 for p in pred[:k]]
    dcg  = sum(r / math.log2(i+2) for i, r in enumerate(rels))
    ideal = sorted(rels, reverse=True)
    idcg  = sum(r / math.log2(i+2) for i, r in enumerate(ideal))
    return dcg/idcg if idcg>0 else 0.0


def evaluate_case(paragraph: str, true_pids: List[str], target_year: Optional[int]=None) -> dict:
    # ─ 1) preprocess & recall ────────────────────────────────────────
    cleaned = clean_text(paragraph)
    chunks  = chunk_tokens(cleaned, TOKENIZER, win=128, stride=64)

    bm25_full = recall_fulltext(cleaned, k=40).assign(source="bm25")
    vec_full  = recall_vector(embed(cleaned), k=40, sim_threshold=0.30).assign(source="vector")
    chunked   = recall_by_chunks(chunks, k_bm25=40, k_vec=40, sim_th=0.30).assign(source="chunked")
    bm25_baseline = recall_fulltext(cleaned, k=20).assign(source="bm25_baseline")

    # candidates for rerankers: concatenate all, dedupe
    all_can = pd.concat([bm25_full, vec_full, chunked], ignore_index=True)
    candidates = (
        all_can.sort_values(["source","sim"], ascending=[True, False])
               .drop_duplicates("pid", keep="first")
               .reset_index(drop=True)
    )

    # ─ 2) metadata & filter ─────────────────────────────────────────
    meta   = fetch_metadata(candidates["pid"].tolist())
    merged = (candidates
              .merge(meta[["pid","abstract","year"]], on="pid", how="left")
              .dropna(subset=["abstract"]))
    if target_year is not None:
        merged = merged[merged["year"] <= target_year]
    if merged.empty:
        raise ValueError("No valid cands after filter")

    # prepare true refs embeddings
    str_refs = [str(p) for p in true_pids]
    ref_meta = fetch_metadata(str_refs).dropna(subset=["abstract"])
    n_rel = len(ref_meta)
    if n_rel:
        ref_embs = np.stack([embed(a) for a in ref_meta["abstract"]])
    else:
        ref_embs = np.zeros((0,768), dtype=float)

    results = {}

    def compute_metrics(predicted: List[str], prefix: str):
        # only keep PIDs that made it through filtering
        valid_pids = [pid for pid in predicted if pid in merged["pid"].values]
        if not valid_pids:
            # no valid candidates → all metrics = 0.0
            for k in TOPK_LIST:
                results[f"{prefix}P@{k}"]    = 0.0
                results[f"{prefix}R@{k}"]    = 0.0
                results[f"{prefix}HR@{k}"]   = 0.0
                results[f"{prefix}NDCG@{k}"] = 0.0
            return

        # build sim_map
        if ref_embs.size:
            # get abstracts in the *filtered* order
            cand_absts = merged.set_index("pid").loc[valid_pids, "abstract"].tolist()
            cand_embs  = np.stack([embed(ab) for ab in cand_absts])
            sims       = cosine_matrix(cand_embs, ref_embs)
            max_sims   = sims.max(axis=1)
        else:
            max_sims = np.zeros(len(valid_pids), dtype=float)

        sim_map = {pid: (sim >= SIM_THRESHOLD)
                   for pid, sim in zip(valid_pids, max_sims)}

        for k in TOPK_LIST:
            results[f"{prefix}P@{k}"]    = precision_by_sim(valid_pids, sim_map, k)
            results[f"{prefix}R@{k}"]    = recall_by_sim(valid_pids, sim_map, k, n_rel)
            results[f"{prefix}HR@{k}"]   = hit_rate_by_sim(valid_pids, sim_map, k)
            results[f"{prefix}NDCG@{k}"] = ndcg_by_sim(valid_pids, sim_map, k)

    # ─ 3) BM25 baseline ───────────────────────────────────────────────
    bm25_pids = bm25_full["pid"].tolist()
    bm25_pids = bm25_baseline["pid"].tolist()
    compute_metrics(bm25_pids, prefix="bm25_")

    # ─ 4) Embedding baseline ──────────────────────────────────────────
    vec_pids = vec_full["pid"].tolist()
    compute_metrics(vec_pids, prefix="vec_")

    # ─ 5) Contextual-prompt LLM rerank ────────────────────────────────
    try:
        rer_ctx = llm_contextual_rerank(
            paragraph,
            merged[["pid","title","abstract"]],
            max_candidates=40,
        )["pid"].tolist()
    except Exception:
        rer_ctx = merged["pid"].tolist()
    compute_metrics(rer_ctx, prefix="ctx_")

    # ─ 6) Loglikelihood LLM rerank ────────────────────────────────────
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

def main(testset_path: Path, max_cases: int):
    df = pd.read_json(testset_path, lines=True).head(max_cases)
    all_metrics = []

    for case in df.to_dict(orient="records"):
        para = case["paragraph"]
        refs = case.get("references", [])
        year = case.get("year") or fetch_year_by_doi(case.get("doi"))
        met = evaluate_case(para, refs, target_year=year)
        met["case_id"] = case.get("id") or case.get("doi")
        all_metrics.append(met)

    res_df = pd.DataFrame(all_metrics).set_index("case_id")
    avg = res_df.mean(numeric_only=True)

    # Prepare comparison table
    def extract(prefix):
        return avg.filter(like=prefix).rename(lambda k: k.replace(prefix, ""))

    comp = pd.DataFrame({
        "bm25":      extract("bm25_"),
        "vector":    extract("vec_"),
        "ctx_llm":   extract("ctx_"),
        "norm_llm":  extract("ll_"),
    })
    print("\nAverage metrics across methods:\n")
    print(comp)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--testset", type=Path, default=default_path)
    p.add_argument("--max_cases", type=int, default=30)
    args = p.parse_args()

    if not args.testset.is_file():
        raise FileNotFoundError(f"Testset not found: {args.testset!r}")
    main(args.testset, args.max_cases)
