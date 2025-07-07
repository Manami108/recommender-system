# user_study_recommender.py
# Revised script: reads a JSONL containing records (with paragraph, name, number, optional id),
# generates top-10 paper recommendations for each non-empty paragraph
# using two reranking methods: (1) RRF + LLM sliding_score and (2) RRF + pure LLaMA scoring,
# and saves the results (including all original fields plus method, rank, title, DOI, year) to a new CSV.

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from chunking import clean_text, chunk_tokens
from recall_user_study import (
    recall_fulltext,
    recall_vector,
    recall_by_chunks,
    rrf_fuse,
    fetch_metadata,
    embed,
)
from rerank_llm import sliding_score, RerankError

# Configuration
INPUT_JSONL    = Path(os.getenv("INPUT_JSONL", "/home/abhi/Desktop/Manami/recommender-system/datasets/user_studies/manami.jsonl"))
OUTPUT_CSV     = Path(os.getenv("OUTPUT_CSV", "/home/abhi/Desktop/Manami/recommender-system/datasets/user_studies/manami_recommendations.csv"))
TOP_K          = int(os.getenv("TOP_K", 15))
MAX_RRF_CAND   = int(os.getenv("MAX_RRF_CAND", 40))
LLAMA_MODEL    = os.getenv("LLAMA_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizers and models
TOKENIZER = AutoTokenizer.from_pretrained(os.getenv("LLM_MODEL", LLAMA_MODEL), use_fast=True)
LLAMA_MODEL_CLS = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL, torch_dtype=torch.float16).to(DEVICE)
LLAMA_MODEL_CLS.eval()


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


def llama_rate(paragraph: str, title: str, abstract: str) -> int:
    prompt = (
        f"Given the research paragraph:\n'''{paragraph}'''\n"
        f"Paper Title: {title}\nAbstract: {abstract}\n"
        "Rate relevance 1 (not relevant) to 5 (highly relevant). Respond with a single digit."
    )
    inputs = TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        out = LLAMA_MODEL_CLS.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=TOKENIZER.pad_token_id
        )
    text = TOKENIZER.decode(out[0], skip_special_tokens=True).strip()
    try:
        score = int(text[0])
    except:
        score = 1
    return max(1, min(5, score))


def recommend_methods(paragraph: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    # Shared RRF pool
    pool = rrf_recall(paragraph, MAX_RRF_CAND)
    pids = pool["pid"].tolist()
    metadata = fetch_metadata(pids)
    # Ensure doi and year columns exist
    if "doi" not in metadata.columns:
        metadata["doi"] = None
    if "year" not in metadata.columns:
        metadata["year"] = None
    # Filter to needed columns
    metadata = metadata[["pid","title","abstract","doi","year"]]

    # Method A: sliding_score rerank
    try:
        df_a = sliding_score(paragraph, metadata[["pid","title","abstract"]])
        df_a = df_a.merge(pool[["pid","rrf_rank"]], on="pid")
        df_a = df_a.sort_values(by=["score","rrf_rank"], ascending=[False,True])
        top_a = df_a.head(TOP_K)["pid"].tolist()
    except RerankError:
        top_a = pool.head(TOP_K)["pid"].tolist()
    final_a = metadata.set_index("pid").loc[top_a].reset_index()
    for rank, row in enumerate(final_a.itertuples(index=False), start=1):
        results.append({
            **{},
            "method": "rrf_llm",
            "rank": rank,
            "title": row.title,
            "doi": row.doi,
            "year": int(row.year) if row.year is not None else None
        })

    # Method B: pure LLaMA rerank
    rates = []
    for pid in pids:
        row_meta = metadata.set_index("pid").loc[pid]
        score = llama_rate(paragraph, row_meta.title, row_meta.abstract)
        rates.append((pid, score))
    df_b = pd.DataFrame(rates, columns=["pid","score"]).merge(pool[["pid","rrf_rank"]], on="pid")
    df_b = df_b.sort_values(by=["score","rrf_rank"], ascending=[False,True])
    top_b = df_b.head(TOP_K)["pid"].tolist()
    final_b = metadata.set_index("pid").loc[top_b].reset_index()
    for rank, row in enumerate(final_b.itertuples(index=False), start=1):
        results.append({
            **{},
            "method": "rrf_pure_llama",
            "rank": rank,
            "title": row.title,
            "doi": row.doi,
            "year": int(row.year) if row.year is not None else None
        })

    return results


def main():
    # Load input JSONL
    records: List[Dict[str, Any]] = []
    with INPUT_JSONL.open('r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if not data.get('paragraph','').strip():
                continue
            records.append(data)

    if not records:
        raise ValueError("No valid paragraphs found.")

    out_rows: List[Dict[str, Any]] = []
    for rec in records:
        base = {k: rec.get(k) for k in rec if k != 'paragraph'}
        recs = recommend_methods(rec['paragraph'])
        for rr in recs:
            out_rows.append({**base, **rr})

    df_out = pd.DataFrame(out_rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ Saved recommendations via both methods to {OUTPUT_CSV} (rows: {len(df_out)})")


if __name__ == '__main__':
    main()
