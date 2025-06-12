#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from chunking import clean_text, chunk_tokens
from recall import (
    recall_by_chunks,
    recall_fulltext,
    recall_vector,
    embed,
    fetch_metadata,
)
from rerank_llm import llm_contextual_rerank
from neo4j import GraphDatabase

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SIM_THRESHOLD = 0.75
TOPK_LIST     = (5, 10, 20)
TOKENIZER     = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASS", "Manami1008")
    )
)

def fetch_year_by_doi(doi: str) -> int | None:
    q = "MATCH (p:Paper {doi: $doi}) RETURN p.year AS year"
    with driver.session() as sess:
        rec = sess.run(q, doi=doi).single()
    return rec["year"] if rec and rec["year"] is not None else None

# ─── METRIC UTILITIES ──────────────────────────────────────────────────────────

def precision_at_k(predicted_pids: list[str], true_pids: list[str], k: int) -> float:
    return len(set(predicted_pids[:k]) & set(true_pids)) / k

def hit_rate_at_k(predicted_pids: list[str], true_pids: list[str], k: int) -> float:
    return float(any(pid in true_pids for pid in predicted_pids[:k]))

# ─── SINGLE‐CASE EVALUATION ────────────────────────────────────────────────────

def evaluate_case(
    paragraph: str,
    true_pids: list[str],
    target_year: int | None = None,
    topk_list: tuple[int,...] = TOPK_LIST
) -> dict[str, float]:
    # 1) preprocess & chunk
    cleaned = clean_text(paragraph)
    chunks  = chunk_tokens(cleaned, TOKENIZER, win=128, stride=64)

    # 2) recall
    bm25_df  = recall_fulltext(cleaned, k=40).assign(source="bm25_full")
    vec_df   = recall_vector(embed(cleaned), k=40, sim_threshold=0.30).assign(source="embed_full")
    chunk_df = recall_by_chunks(chunks,  k_bm25=40, k_vec=40, sim_th=0.30).assign(source="chunked")

    # 2b) merge & dedupe
    recs = pd.concat([bm25_df, vec_df, chunk_df], ignore_index=True)
    recs = (
        recs
        .sort_values(["source","sim"], ascending=[True, False])
        .drop_duplicates("pid", keep="first")
        .reset_index(drop=True)
    )

    # 3) fetch metadata & filter abstracts
    meta = fetch_metadata(recs["pid"].tolist())
    merged = recs.merge(meta[["pid","abstract","year"]], on="pid", how="left")
    merged = merged[merged["abstract"].notna()]

    # 4) year filter
    if target_year is not None:
        merged = merged[merged["year"] <= target_year]
    if merged.empty:
        raise ValueError(f"No candidates ≤ year {target_year}")

    # 5) LLM‐based rerank
    try:
        rerank_df = llm_contextual_rerank(
            paragraph,
            merged[["pid","title","abstract"]],
            max_candidates=40
        )
        predicted = [pid for pid in rerank_df["pid"] if pid in merged["pid"].tolist()]
    except Exception:
        predicted = merged.sort_values("sim", ascending=False)["pid"].tolist()

    # 6) compute metrics
    results: dict[str, float] = {}
    for k in topk_list:
        results[f"P@{k}"]  = precision_at_k(predicted, true_pids, k)
        results[f"HR@{k}"] = hit_rate_at_k(predicted, true_pids, k)
    return results

# ─── MAIN LOOP ─────────────────────────────────────────────────────────────────

def main(testset_path: str, max_cases: int = 20):
    if not os.path.isfile(testset_path):
        raise FileNotFoundError(f"Testset not found: {testset_path}")

    # read & limit
    df = pd.read_json(testset_path, lines=True).head(max_cases)
    all_metrics: list[dict[str, float|str]] = []

    for case in df.to_dict(orient="records"):
        para = case["paragraph"]
        # make sure references are strings
        true_pids = [str(pid) for pid in case.get("references", [])]
        case_id   = case.get("id") or case.get("doi")
        year      = case.get("year") or fetch_year_by_doi(case.get("doi"))

        metrics = evaluate_case(para, true_pids, target_year=year)
        metrics["case_id"] = case_id
        all_metrics.append(metrics)

    # build results DataFrame
    res_df = pd.DataFrame(all_metrics).set_index("case_id")
    print("Per‐case metrics:\n", res_df)
    print("\nAverage metrics (numeric only):\n", res_df.mean(numeric_only=True))

if __name__ == "__main__":
    path = os.getenv(
        "TESTSET_PATH",
        "/home/abhi/Desktop/Manami/recommender-system/datasets/testset_300_references.jsonl"
    )
    main(path)
