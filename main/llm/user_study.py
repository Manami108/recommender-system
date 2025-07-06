# user_study_recommender.py
# This script reads a JSONL containing paragraphs, generates top-10 paper recommendations for each
# paragraph using RRF + LLM reranking, and saves the results (title, DOI, year) to a new JSONL.

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
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
from rerank_llm import sliding_score, RerankError

# Configuration
INPUT_JSONL  = Path(os.getenv("INPUT_JSONL", "online.jsonl"))
OUTPUT_JSONL = Path(os.getenv("OUTPUT_JSONL", "online_recommendations.jsonl"))
TOP_K        = int(os.getenv("TOP_K", 10))
MAX_RRF_CAND = int(os.getenv("MAX_RRF_CAND", 40))

# Tokenizer for chunking (used in retrieval)
TOKENIZER = AutoTokenizer.from_pretrained(
    os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    use_fast=True
)

# Cosine similarity on embeddings
def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T


def recommend_topk(
    paragraph: str,
    top_k: int = TOP_K,
    max_rrf: int = MAX_RRF_CAND
) -> pd.DataFrame:
    """
    Generate top-k recommendations for a given paragraph.
    Returns a DataFrame with columns [paragraph_id, rank, pid, title, doi, year] for the top-k papers.
    """
    # 1. Text cleanup and chunking
    cleaned = clean_text(paragraph)
    chunks = chunk_tokens(cleaned, TOKENIZER, win=256, stride=64)

    # 2. Retrieve via BM25 & vector
    full_bm25 = recall_fulltext(cleaned).assign(src="full_bm25")
    full_vec  = recall_vector(embed(cleaned)).assign(src="full_vec")
    chunk_pool = recall_by_chunks(chunks)

    # Prepare for RRF fusion: keep only pid & rank
    for df in (full_bm25, full_vec):
        df.drop(columns=[c for c in df.columns if c not in ("pid","rank")], inplace=True)

    # 3. RRF fusion
    pool = rrf_fuse(full_bm25, full_vec, chunk_pool, top_k=max_rrf).reset_index(drop=True)
    pool["rrf_rank"] = np.arange(len(pool))

    # 4. Fetch metadata
    cand = fetch_metadata(pool["pid"].tolist())
    if cand.empty:
        raise RuntimeError("No candidates returned after metadata fetch.")

    # 5. LLM reranking
    try:
        llm_df = sliding_score(paragraph, cand[["pid", "title", "abstract"]])
        llm_df = llm_df.merge(pool[["pid","rrf_rank"]], on="pid", how="left")
        llm_df = llm_df.sort_values(by=["score","rrf_rank"], ascending=[False,True])
        top_pids = llm_df["pid"].tolist()[:top_k]
    except RerankError:
        top_pids = pool["pid"].tolist()[:top_k]

    # 6. Assemble final metadata
    recs = fetch_metadata(top_pids)
    recs = recs.set_index("pid").loc[top_pids].reset_index()
    recs = recs[["title","doi","year"]]
    return recs


def main():
    # Load input JSONL (expects fields 'paragraph' and optional 'id')
    records = []
    with INPUT_JSONL.open('r') as f:
        for line in f:
            records.append(json.loads(line))
    if not all('paragraph' in rec for rec in records):
        raise ValueError("Each JSON line must contain a 'paragraph' field.")

    output_lines = []
    for rec in records:
        para_id = rec.get('id')
        paragraph = rec['paragraph']
        recs = recommend_topk(paragraph)
        for rank, row in recs.iterrows():
            output_record = {
                'paragraph_id': para_id,
                'rank': rank + 1,
                'title': row['title'],
                'doi': row['doi'],
                'year': int(row['year']),
            }
            output_lines.append(output_record)

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    # Write output JSONL
    with OUTPUT_JSONL.open('w') as f:
        for out in output_lines:
            f.write(json.dumps(out) + '\n')
    print(f"✅ Saved top-{TOP_K} recommendations for {len(records)} paragraphs to {OUTPUT_JSONL}")

if __name__ == '__main__':
    main()
