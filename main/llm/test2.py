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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # assumes embeddings are normalized

# ─── Neo4j Driver setup ─────────────────────────────────────────────────────────

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASS", "Manami1008"))
)


def fetch_year_by_doi(doi: str) -> int | None:
    q = "MATCH (p:Paper {doi: $doi}) RETURN p.year AS year"
    with driver.session() as sess:
        rec = sess.run(q, doi=doi).single()
    return rec["year"] if rec else None

# ─── Tokenizer & config ─────────────────────────────────────────────────────────

TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
SIM_THRESHOLD = 0.75  # similarity threshold for reference matching

# ─── Single-case evaluation ────────────────────────────────────────────────────

def evaluate_case(paragraph, true_pids, target_year=None, topk_list=(5,10,20)):
    # 1) Preprocess & chunk
    cleaned = clean_text(paragraph)
    chunks = chunk_tokens(cleaned, TOKENIZER, win=128, stride=64)

    # 2a) Global/full-text recall
    full_bm25 = recall_fulltext(cleaned, k=40).assign(source='bm25_full')
    full_vec  = recall_vector(embed(cleaned), k=40, sim_threshold=0.30).assign(source='embed_full')
    # 2b) Chunk-based recall
    chunk_df  = recall_by_chunks(chunks, k_bm25=40, k_vec=40, sim_th=0.30).assign(source='chunked')

    # 2c) Combine & dedupe
    rec_df = pd.concat([full_bm25, full_vec, chunk_df], ignore_index=True)
    rec_df = (
        rec_df
        .sort_values(['source','sim'], ascending=[True, False])
        .drop_duplicates('pid', keep='first')
        .reset_index(drop=True)
    )

    # 3) Fetch metadata (title, abstract, year)
    rec_core = rec_df[['pid','sim','source','hop']] if 'hop' in rec_df.columns else rec_df[['pid','sim','source']]
    meta_rec = fetch_metadata(rec_core['pid'].tolist())
    df_rec = rec_core.merge(meta_rec[['pid','title','abstract','year']], on='pid', how='left')

    # 4) Filter by year
    if target_year is not None:
        df_rec = df_rec[df_rec['year'] <= target_year]
    if df_rec.empty:
        raise ValueError(f"No valid candidates ≤ year {target_year}")

    # 5) LLM rerank (fallback to sim)
    try:
        reranked = llm_contextual_rerank(
            paragraph,
            df_rec[['pid','title','abstract']],
            max_candidates=40
        )
        predicted = reranked['pid'].tolist()
    except Exception:
        predicted = df_rec.sort_values('sim', ascending=False)['pid'].tolist()

    # 6) Compute precision and hit-rate metrics
    metrics = {}
    for k in topk_list:
        metrics[f"P@{k}"] = precision_at_k(predicted, true_pids, k)
        metrics[f"HR@{k}"] = hit_rate_at_k(predicted, true_pids, k)

    # 7) Reference similarity check
    # 7a) Fetch reference abstracts and embeddings
    meta_true = fetch_metadata(true_pids)
    ref_embeddings = {}
    for pid, abst in zip(meta_true['pid'], meta_true['abstract']):
        if isinstance(abst, str) and abst.strip():
            ref_embeddings[pid] = np.array(embed(abst))
    # 7b) Evaluate each recommended paper against all references
    sim_results = []
    for pid in predicted:
        row = df_rec[df_rec['pid'] == pid]
        abst = row['abstract'].iloc[0] if 'abstract' in row else None
        if not isinstance(abst, str) or not abst.strip():
            sim_results.append((pid, None, False))
            continue
        emb_rec = np.array(embed(abst))
        # compute max cosine across all refs
        max_sim = 0.0
        for ref_emb in ref_embeddings.values():
            s = cosine_similarity(emb_rec, ref_emb)
            if s > max_sim:
                max_sim = s
        correct = (max_sim >= SIM_THRESHOLD)
        sim_results.append((pid, max_sim, correct))

    # 8) Attach similarity results to metrics
    metrics['similarity_results'] = sim_results
    return metrics

# ─── Main evaluation loop ──────────────────────────────────────────────────────

def main(testset_path, max_cases=):
    df = pd.read_json(testset_path, lines=True)
    df = df.head(max_cases)
    cases = df.to_dict(orient='records')

    all_results = []
    for case in cases:
        paragraph = case['paragraph']
        true_pids = case['references']
        case_id    = case.get('id') or case.get('doi')
        target_year= case.get('year') or fetch_year_by_doi(case.get('doi'))

        res = evaluate_case(paragraph, true_pids, target_year)
        res['case_id'] = case_id
        all_results.append(res)

    # Build a summary DataFrame
    rows = []
    for r in all_results:
        base = {k: v for k, v in r.items() if k.startswith('P@') or k.startswith('HR@')}
        base['case_id'] = r['case_id']
        rows.append(base)
    summary = pd.DataFrame(rows).set_index('case_id')

    print("Per-case metrics:\n", summary)
    print("\nAverage metrics:\n", summary.mean())
    # Optionally, print similarity details for the first case
    print("\nSample similarity results for first case:")
    print(all_results[0]['similarity_results'])

if __name__ == '__main__':
    path = os.getenv('TESTSET_PATH', 'datasets/testset_300_references.jsonl')
    main(path)
