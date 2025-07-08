from __future__ import annotations
import faulthandler
faulthandler.enable(all_threads=True, file=open("fault.log", "w"))

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List
from rerank_llm import sliding_score, RerankError
from chunking import clean_text, chunk_tokens
from recall_user_study import (
    recall_fulltext,
    recall_vector,
    recall_by_chunks,
    rrf_fuse,
    fetch_metadata,
    embed,
)

# Config
INPUT_JSONL = Path(os.getenv("INPUT_JSONL", "/home/abhi/Desktop/Manami/recommender-system/datasets/user_studies/dewa.jsonl"))
OUTPUT_CSV  = Path(os.getenv("OUTPUT_CSV", "/home/abhi/Desktop/Manami/recommender-system/datasets/user_studies/dewa_recommendations.csv"))
TOP_K       = int(os.getenv("TOP_K", 15))
MAX_RRF_CAND= int(os.getenv("MAX_RRF_CAND", 40))

# Tokenizer (shared)
from transformers import AutoTokenizer
TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", use_fast=True)

# RRF retrieval

def rrf_recall(paragraph: str, max_rrf: int) -> pd.DataFrame:
    clean = clean_text(paragraph)
    chunks = chunk_tokens(clean, TOKENIZER, win=256, stride=64)
    bm25 = recall_fulltext(clean).assign(src="full_bm25")
    vec  = recall_vector(embed(clean)).assign(src="full_vec")
    chunk_pool = recall_by_chunks(chunks)
    for df in (bm25, vec):
        df.drop(columns=[c for c in df.columns if c not in ("pid","rank")], inplace=True)
    pool = rrf_fuse(bm25, vec, chunk_pool, top_k=max_rrf).reset_index(drop=True)
    pool["rrf_rank"] = np.arange(len(pool))
    return pool

# Recommend

def recommend_rrf_llm(paragraph: str) -> List[Dict[str, Any]]:
    pool = rrf_recall(paragraph, MAX_RRF_CAND)
    pids = pool["pid"].tolist()
    meta = fetch_metadata(pids)[["pid","title","abstract","doi","year"]]

    # sliding_score rerank
    try:
        df = sliding_score(paragraph, meta[["pid","title","abstract"]])
        df = df.merge(pool[["pid","rrf_rank"]], on="pid")
        df = df.sort_values(by=["score","rrf_rank"], ascending=[False,True])
        top = df.head(TOP_K)["pid"].tolist()
    except RerankError:
        top = pool.head(TOP_K)["pid"].tolist()

    final = meta.set_index("pid").loc[top].reset_index()
    return [{
        **{},
        "method": "rrf_llm",
        "rank": i+1,
        "title": row.title,
        "doi": row.doi,
        "year": int(row.year) if row.year is not None else None
    } for i, row in enumerate(final.itertuples(index=False))]

# Main

def main():
    records = []
    with INPUT_JSONL.open('r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data.get('paragraph','').strip(): records.append(data)
    out = []
    for rec in records:
        base = {k: rec[k] for k in rec if k!='paragraph'}
        recs = recommend_rrf_llm(rec['paragraph'])
        for r in recs:
            out.append({**base, **r})
    pd.DataFrame(out).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ rrf_llm recommendations saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
