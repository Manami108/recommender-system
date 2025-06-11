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

# ─── Metric utilities ──────────────────────────────────────────────────────────

def precision_at_k(predicted_pids, true_pids, k):
    topk = predicted_pids[:k]
    return len(set(topk) & set(true_pids)) / k


def hit_rate_at_k(predicted_pids, true_pids, k):
    return float(any(pid in true_pids for pid in predicted_pids[:k]))

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASS", "Manami1008"))
)

def fetch_year_by_doi(doi: str) -> int | None:
    q = "MATCH (p:Paper {doi: $doi}) RETURN p.year AS year"
    with driver.session() as sess:
        rec = sess.run(q, doi=doi).single()
    return rec["year"] if rec else None


# ─── Pre-load tokenizer once ───────────────────────────────────────────────────
TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# ─── Single-case evaluation ────────────────────────────────────────────────────

def evaluate_case(paragraph, true_pids, target_year=None, topk_list=(5,10,20)):
    # 1) Preprocess & chunk
    cleaned = clean_text(paragraph)
    chunks = chunk_tokens(cleaned, TOKENIZER, win=128, stride=64)

    # 2a) Full-text recall
    full_bm25 = recall_fulltext(cleaned, k=40).assign(source='bm25_full')
    full_vec = recall_vector(embed(cleaned), k=40, sim_threshold=0.30).assign(source='embed_full')

    # 2b) Chunk-based recall
    chunk_df = recall_by_chunks(chunks, k_bm25=40, k_vec=40, sim_th=0.30).assign(source='chunked')

    # 2c) Combine & dedupe
    rec_df = pd.concat([full_bm25, full_vec, chunk_df], ignore_index=True)
    rec_df = (
        rec_df
        .sort_values(['source','sim'], ascending=[True, False])
        .drop_duplicates('pid', keep='first')
        .reset_index(drop=True)
    )

    # 3) Fetch metadata & prepare for rerank
    meta = fetch_metadata(rec_df['pid'].tolist())
    df_merge = rec_df.merge(meta[['pid','title','abstract','year']], on='pid', how='left')
    if target_year is not None:
        df_merge = df_merge[df_merge['year'] <= target_year]
    if df_merge.empty:
        raise ValueError(f"No valid candidates ≤ year {target_year}")

    # 4) LLaMA contextual coherence rerank with fallback
    try:
        reranked = llm_contextual_rerank(
            paragraph,
            df_merge[['pid','title','abstract']],
            max_candidates=40  # limit for memory efficiency
        )
        predicted = reranked['pid'].tolist()
    except ValueError as e:
        print(f"LLM rerank failed ({e}); falling back to sim ranking.")
        # fallback: use combined sim sorting
        predicted = rec_df.sort_values('sim', ascending=False)['pid'].tolist()

    # 5) Compute metrics
    results = {}
    for k in topk_list:
        results[f"P@{k}"] = precision_at_k(predicted, true_pids, k)
        results[f"HR@{k}"] = hit_rate_at_k(predicted, true_pids, k)
    return results

# ─── Main eval loop ───────────────────────────────────────────────────────────

def main(testset_path, max_cases=20):
    # Read JSONL via pandas
    
    df = pd.read_json(testset_path, lines=True)
    df = df.head(max_cases)
    cases = df.to_dict(orient='records')

    
    all_metrics = []
    for case in cases:
        paragraph = case['paragraph']
        true_pids = case['references']
        case_id   = case.get('id') or case.get('doi')
        target_year = case.get('year') or fetch_year_by_doi(case.get('doi'))
        metrics   = evaluate_case(paragraph, true_pids, target_year)
        metrics['case_id'] = case_id
        all_metrics.append(metrics)

    result_df = pd.DataFrame(all_metrics).set_index('case_id')
    print("Per-case results:\n", result_df)
    print("\nAverage across all cases:\n", result_df.mean())

if __name__ == '__main__':
    path = os.getenv('TESTSET_PATH',
                     '/home/abhi/Desktop/Manami/recommender-system/datasets/testset_300_references.jsonl')
    main(path)
