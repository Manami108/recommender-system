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
NEO4J_URI     = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER    = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASS    = os.getenv('NEO4J_PASS', 'Manami1008')
SIM_THRESHOLD = 0.75  # embedding similarity cutoff for correctness
TOPK_LIST     = (5, 10, 20)
TOKENIZER     = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# ─── NEO4J DRIVER ──────────────────────────────────────────────────────────────
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def fetch_year_by_doi(doi: str) -> int | None:
    q = "MATCH (p:Paper {doi:$doi}) RETURN p.year AS year"
    with driver.session() as sess:
        rec = sess.run(q, doi=doi).single()
    return rec['year'] if rec and rec['year'] is not None else None

# ─── METRIC UTILITIES ─────────────────────────────────────────────────────────
def precision_by_sim(pred: list[str], sim_map: dict[str,bool], k: int) -> float:
    flags = [sim_map.get(pid, False) for pid in pred[:k]]
    return sum(flags) / k if k else 0.0

def hit_rate_by_sim(pred: list[str], sim_map: dict[str,bool], k: int) -> float:
    return float(any(sim_map.get(pid, False) for pid in pred[:k]))

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # assume normalized embeddings

# ─── EVALUATION ────────────────────────────────────────────────────────────────
def evaluate_case(paragraph: str, true_pids: list[str], target_year: int | None = None) -> dict:
    # 1) Preprocess & chunk
    cleaned = clean_text(paragraph)
    chunks  = chunk_tokens(cleaned, TOKENIZER, win=128, stride=64)

    # 2) Recall candidates
    bm25_full = recall_fulltext(cleaned, k=40).assign(source='bm25_full')
    vec_full  = recall_vector(embed(cleaned), k=40, sim_threshold=0.30).assign(source='embed_full')
    chunked   = recall_by_chunks(chunks, k_bm25=40, k_vec=40, sim_th=0.30).assign(source='chunked')
    candidates = pd.concat([bm25_full, vec_full, chunked], ignore_index=True)
    candidates = candidates.sort_values(['source','sim'], ascending=[True, False])
    candidates = candidates.drop_duplicates('pid', keep='first').reset_index(drop=True)

    # 3) Fetch metadata (title, abstract, year) and filter out missing abstracts
    meta = fetch_metadata(candidates['pid'].tolist())
    merged = candidates.merge(meta[['pid','abstract','year']], on='pid', how='left')
    merged = merged[merged['abstract'].notna()]

    # 4) Year filter and ensure we still have abstracts
    if target_year is not None:
        merged = merged[merged['year'] <= target_year]
    if merged.empty:
        raise ValueError("No valid candidates with abstracts after filtering by year and availability")

    # 5) Rerank with LLM (only on items that have abstracts)
    try:
        reranked = llm_contextual_rerank(
            paragraph,
            merged[['pid','title','abstract']],
            max_candidates=40
        )
        # ensure we only keep reranker outputs that had abstracts
        predicted = [pid for pid in reranked['pid'] if pid in merged['pid'].tolist()]
    except Exception:
        predicted = merged['pid'].tolist()

    # 6) Prepare reference embeddings (skip refs without abstracts)
    ref_meta = fetch_metadata(true_pids)
    ref_meta = ref_meta[ref_meta['abstract'].notna()]
    ref_embs = [np.array(embed(ab)) for ab in ref_meta['abstract'].tolist()]

    # 7) Build similarity map for predicted
    abst_map = dict(zip(merged['pid'], merged['abstract']))
    sim_map = {}
    for pid in predicted:
        abst = abst_map.get(pid)
        if not abst:
            sim_map[pid] = False
            continue
        rec_emb = np.array(embed(abst))
        max_sim = max((cosine_similarity(rec_emb, r) for r in ref_embs), default=0.0)
        sim_map[pid] = (max_sim >= SIM_THRESHOLD)

    # 8) Compute similarity-based metrics
    results = {}
    for k in TOPK_LIST:
        results[f"P@{k}"] = precision_by_sim(predicted, sim_map, k)
        results[f"HR@{k}"] = hit_rate_by_sim(predicted, sim_map, k)
    results['sim_map'] = sim_map
    return results

# ─── MAIN LOOP ─────────────────────────────────────────────────────────────────
def main(testset_path: str, max_cases: int = 2):
    if not os.path.isfile(testset_path):
        raise FileNotFoundError(f"Testset file not found: {testset_path}")
    df = pd.read_json(testset_path, lines=True).head(max_cases)

    all_metrics = []
    for case in df.to_dict(orient='records'):
        para = case['paragraph']
        refs = case.get('references', [])
        cid  = case.get('id') or case.get('doi')
        year = case.get('year') or fetch_year_by_doi(case.get('doi'))
        metrics = evaluate_case(para, refs, target_year=year)
        metrics['case_id'] = cid
        all_metrics.append(metrics)

    res_df = pd.DataFrame(all_metrics).set_index('case_id')
    print("Per-case metrics:\n", res_df)
    print("\nAverage metrics:\n", res_df.mean())

if __name__ == '__main__':
    path = os.getenv('TESTSET_PATH', '/home/abhi/Desktop/Manami/recommender-system/datasets/testset_300_references.jsonl')
    main(path)
