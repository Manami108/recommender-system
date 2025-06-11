import os
import json
import numpy as np
import pandas as pd
from numpy.linalg import norm
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
def precision_at_k(relevant_flags: np.ndarray, k: int) -> float:
    return float(relevant_flags[:k].sum()) / k

def hit_rate_at_k(relevant_flags: np.ndarray, k: int) -> float:
    return float(bool(relevant_flags[:k].any()))

# ─── Cosine similarity ─────────────────────────────────────────────────────────
def cosine(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v) / (norm(u) * norm(v)))

# ─── Neo4j driver for DOI → year lookup ────────────────────────────────────────
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
def evaluate_case(
    paragraph: str,
    true_pids: list[int],
    target_year: int | None = None,
    topk_list: tuple[int,...] = (5,10,20),
    rel_threshold: float = 0.75
) -> dict[str, float]:
    # 1) Preprocess & chunk
    cleaned = clean_text(paragraph)
    chunks = chunk_tokens(cleaned, TOKENIZER, win=128, stride=64)

    # 2a) Full-text & vector recall
    full_bm25 = recall_fulltext(cleaned, k=100).assign(source='bm25_full')
    full_vec  = recall_vector(embed(cleaned), k=100, sim_threshold=0.0).assign(source='embed_full')

    # 2b) Chunk-based recall
    chunk_df  = recall_by_chunks(chunks, k_bm25=100, k_vec=100, sim_th=0.0).assign(source='chunked')

    # 2c) Combine & dedupe
    rec_df = pd.concat([full_bm25, full_vec, chunk_df], ignore_index=True)
    rec_df = (
        rec_df
        .sort_values(['source','sim'], ascending=[True, False])
        .drop_duplicates('pid', keep='first')
        .reset_index(drop=True)
    )

    # 3) Fetch metadata & filter by year
    rec_core = rec_df[['pid','sim','source']]
    meta = fetch_metadata(rec_core['pid'].tolist())
    df_merge = rec_core.merge(meta[['pid','title','abstract','year']], on='pid', how='left')
    if target_year is not None:
        df_merge = df_merge[df_merge['year'] <= target_year]
    if df_merge.empty:
        raise ValueError(f"No valid candidates ≤ year {target_year}")

    # 4) Final recommendation list (use LLM fallback if desired)
    try:
        reranked = llm_contextual_rerank(
            paragraph,
            df_merge[['pid','title','abstract']],
            max_candidates=40
        )
        predicted_pids = reranked['pid'].tolist()
    except Exception:
        predicted_pids = df_merge.sort_values('sim', ascending=False)['pid'].tolist()

    # 5) Compute relevance flags: candidate relevant if sim to ANY true-citation >= threshold
    # 5a) get abstracts for predictions and true citations
    pred_meta  = fetch_metadata(predicted_pids)
    true_meta  = fetch_metadata(true_pids)
    pred_abs   = pred_meta.set_index('pid').loc[predicted_pids, 'abstract'].fillna("").tolist()
    true_abs   = true_meta.set_index('pid').loc[true_pids, 'abstract'].fillna("").tolist()
    # 5b) embed both lists in batches
    pred_embs = np.stack([embed(txt) for txt in pred_abs])
    true_embs = np.stack([embed(txt) for txt in true_abs])
    # 5c) similarity matrix and relevance vector
    sim_matrix = pred_embs @ true_embs.T
    relevant_flags = (sim_matrix.max(axis=1) >= rel_threshold)

    # 6) Compute P@K and HR@K on relevance flags
    results = {}
    for k in topk_list:
        results[f"P@{k}"]  = precision_at_k(relevant_flags, k)
        results[f"HR@{k}"] = hit_rate_at_k(relevant_flags, k)
    return results

# ─── Main eval loop ───────────────────────────────────────────────────────────
def load_jsonl(path: str) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)


def main(testset_path: str, max_cases: int = 20):
    df = load_jsonl(testset_path).head(max_cases)
    cases = df.to_dict(orient='records')

    all_metrics = []
    for case in cases:
        paragraph   = case['paragraph']
        true_pids   = case['references']
        case_id     = case.get('id') or case.get('doi')
        target_year = case.get('year') or fetch_year_by_doi(case.get('doi'))
        metrics     = evaluate_case(paragraph, true_pids, target_year)
        metrics['case_id'] = case_id
        all_metrics.append(metrics)

    result_df = pd.DataFrame(all_metrics).set_index('case_id')
    print("Per-case results:\n", result_df)
    print("\nAverage across all cases:\n", result_df.mean())

if __name__ == '__main__':
    path = os.getenv('TESTSET_PATH',
                     '/home/abhi/Desktop/Manami/recommender-system/datasets/testset_300_references.jsonl')
    main(path)
