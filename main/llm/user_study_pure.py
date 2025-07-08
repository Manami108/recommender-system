# llm_pure _reranking.py
# Performs RRF fusion followed by per-candidate pure LLaMA scoring

from __future__ import annotations
import faulthandler
faulthandler.enable(all_threads=True, file=open("fault.log", "w"))

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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
INPUT_JSONL = Path(os.getenv("INPUT_JSONL", "/home/abhi/Desktop/Manami/recommender-system/datasets/user_studies/manami.jsonl"))
OUTPUT_CSV  = Path(os.getenv("OUTPUT_CSV", "/home/abhi/Desktop/Manami/recommender-system/datasets/user_studies/manami_recommendations2.csv"))
TOP_K       = int(os.getenv("TOP_K", 15))
MAX_RRF_CAND= int(os.getenv("MAX_RRF_CAND", 40))
# LLaMA model & TOKENIZER
MODEL_NAME = os.getenv("llm_pure _MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
TOKENIZER  = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
llm_pure      = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).cuda()
llm_pure .eval()
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RRF recall (same as above)
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

# pure-LLM rating function
def llama_rate(paragraph: str, title: str, abstract: str) -> int:
    prompt = (
        f"Given the research paragraph:\n'''{paragraph}'''\n"
        f"Paper Title: {title}\nAbstract: {abstract}\n"
        "Rate relevance 1 (not relevant) to 5 (highly relevant). Respond with a single digit."
    )
    inputs = TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        out = llm_pure .generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=TOKENIZER.pad_token_id
        )
    txt = TOKENIZER.decode(out[0], skip_special_tokens=True).strip()
    try:
        return max(1, min(5, int(txt[0])))
    except:
        return 1

# Recommend
def recommend_rrf_llm_pure (paragraph: str) -> List[Dict[str, Any]]:
    pool = rrf_recall(paragraph, MAX_RRF_CAND)
    pids = pool["pid"].tolist()
    meta = fetch_metadata(pids)[["pid","title","abstract","doi","year"]]

    rates = []
    for pid in pids:
        row = meta.set_index("pid").loc[pid]
        rates.append((pid, llama_rate(paragraph, row.title, row.abstract)))
    df = pd.DataFrame(rates, columns=["pid","score"]).merge(pool[["pid","rrf_rank"]], on="pid")
    df = df.sort_values(by=["score","rrf_rank"], ascending=[False,True])
    top = df.head(TOP_K)["pid"].tolist()

    final = meta.set_index("pid").loc[top].reset_index()
    return [{
        **{},
        "method": "rrf_llm_pure ",
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
            d = json.loads(line)
            if d.get('paragraph','').strip(): records.append(d)
    out = []
    for rec in records:
        base = {k: rec[k] for k in rec if k!='paragraph'}
        recs = recommend_rrf_llm_pure (rec['paragraph'])
        for r in recs:
            out.append({**base, **r})
    pd.DataFrame(out).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ pure-llama recommendations saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
