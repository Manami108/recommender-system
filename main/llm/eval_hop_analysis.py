# eval_rrf_llm_hop.py  (20 RRF seeds → hop → LLM top-20)

from __future__ import annotations
import os, math, logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from chunking import clean_text, chunk_tokens
from recall   import (
    recall_fulltext,
    recall_vector,
    recall_by_chunks,
    rrf_fuse,
    fetch_metadata,
    embed,
)
from rerank_llm    import rerank_batch, RerankError
from hop_reasoning import multi_hop_topic_citation_reasoning   # <<< hop >>>

# config
TESTSET_PATH = Path(os.getenv(
    "TESTSET_PATH",
    "/home/abhi/Desktop/Manami/recommender-system/datasets/testset_2020_references.jsonl"))
MAX_CASES  = int(os.getenv("MAX_CASES", 2))
TOPK_LIST  = tuple(range(1, 21))
SIM_THRESH = 0.95

RRF_TOPK       = 20  # keep top 20 seeds after RRF fusion
# HOP_TOP_N      = 3   # retrieve up to 20 hop papers per seed
FINAL_POOL_CAP = 60  # cap total pool size before final LLM
LLM_TOPK       = 20  # final list size

TOKENIZER = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct", use_fast=True)

# cosin similarity 
def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T

# 
def evaluate_case(
    paragraph: str,
    gold_pids: List[str],
    hop_top_n: int,  
    final_k:    int,                         # required first
    target_year: Optional[int] = None,   # defaulted second
) -> dict:

    # take the raw paragraph, clean it (lower-casing, removing punctuation/stopwords, etc.), 
    # so that BM25 and embeddings both work on normalized text.
    clean = clean_text(paragraph)
    chunks = chunk_tokens(clean, TOKENIZER, win=256, stride=64)

    # recall & fuse with reciprocal rank fusion (RRF)
    # full-text & vector for whole paragraph
    bm25 = recall_fulltext(clean).assign(src="full_bm25")
    vec  = recall_vector(embed(clean)).assign(src="full_vec")
    chunk_pool = recall_by_chunks(chunks)

    # apply RRF fusion across all sources
    # full-doc scores (bm25_score, semantic_score) → hybrid_full
    # chunk-level max scores → hybrid_chunk
    # Total hybrid_score = hybrid_full + hybrid_chunk
    # strip out any extraneous columns so rrf_fuse sees just pid & rank
    # strip everything except pid & rank on the two full-doc tables
    for df in (bm25, vec):
        df.drop(columns=[c for c in df.columns if c not in ("pid", "rank")], inplace=True)

    seeds_df = rrf_fuse(bm25, vec, chunk_pool, top_k=RRF_TOPK)
    seed_ids = seeds_df.pid.tolist()

    # this is hop expansions
    hop_df    = multi_hop_topic_citation_reasoning(seed_ids, top_n=hop_top_n)
    union_ids = list(dict.fromkeys(seed_ids + hop_df.pid.tolist()))[:FINAL_POOL_CAP]  # <<< hop >>>
    meta_pool = fetch_metadata(union_ids).dropna(subset=["abstract"])
    if target_year is not None:
        meta_pool = meta_pool[meta_pool.year < target_year]

    #  rerank via LLM scoring
    try:
        reranked = rerank_batch(
            paragraph,
            meta_pool[["pid", "title", "abstract"]],
            k=final_k,
            max_candidates=len(meta_pool))
        final_ids = reranked.pid.tolist()
    except RerankError as e:
        logging.warning("LLM rerank failed → fallback order (%s)", e)
        final_ids = meta_pool.pid.head(LLM_TOPK).tolist()

    # embed references & predicted candidates
    # Calculates cosine similarities between reranked abstracts and true references.
    # For each predicted paper, checks if any unmatched reference has sim ≥ 0.95 → it's a hit.
    refs     = fetch_metadata(gold_pids).dropna(subset=["abstract"])
    ref_ids  = refs.pid.tolist()
    ref_emb  = np.stack([embed(a) for a in refs.abstract]) if ref_ids else np.zeros((0,768))
    cand_emb = np.stack([embed(a) for a in meta_pool.set_index("pid").loc[final_ids].abstract])

    # greedy match for hit detection
    sims = cosine_matrix(cand_emb, ref_emb) if ref_ids else np.zeros((len(final_ids),0))
    unmatched, hits = set(range(len(ref_ids))), []
    for i in range(len(final_ids)):
        if not unmatched: hits.append(False); continue
        idxs = list(unmatched)
        j = sims[i, idxs].argmax()
        if sims[i, idxs][j] >= SIM_THRESH:
            hits.append(True); unmatched.remove(idxs[j])
        else:
            hits.append(False)

    # https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems    
    # P@k  = # relevant hits in top-k / k
    # HR@k = # whether at least one hit in top-k
    # R@k  = # hits / total ground-truth
    # NDCG = # relevance-weighted discounted ranking quality
    out, n_rel = {}, len(ref_ids)
    for k in TOPK_LIST:
        topk = hits[:k]
        out[f"P@{k}"]  = sum(topk)/k
        out[f"HR@{k}"] = float(any(topk))
        out[f"R@{k}"]  = sum(topk)/n_rel if n_rel else 0.0
        dcg  = sum(r/math.log2(i+2) for i,r in enumerate(topk))
        idcg = sum(1/math.log2(i+2) for i in range(min(n_rel,k)))
        out[f"NDCG@{k}"] = dcg/idcg if idcg else 0.0
    return out

import matplotlib
matplotlib.use("Agg")          # off-screen backend

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")

    df = pd.read_json(TESTSET_PATH, lines=True).head(MAX_CASES)

    hop_values   = [1, 3, 5, 10]      # sweep this
    rec_k_values = [3, 5, 10, 20]     # just for plotting
    sweep_rows   = []                 # will hold one row per hop_n

    for hop_n in hop_values:
        per_paragraph = []
        for rec in df.to_dict("records"):
            m = evaluate_case(
                paragraph   = rec["paragraph"],
                gold_pids   = [str(pid) for pid in rec.get("references", [])],
                hop_top_n   = hop_n,
                final_k     = 20,            # always generate 20
                target_year = rec.get("year")
            )
            per_paragraph.append(m)

        avg = pd.DataFrame(per_paragraph).mean(numeric_only=True)

        # extract just the ks we care about
        sweep_rows.append({
            "hop_n":   hop_n,
            "P@3":     avg["P@3"],
            "P@5":     avg["P@5"],
            "P@10":    avg["P@10"],
            "P@20":    avg["P@20"],
            "HR@3":    avg["HR@3"],
            "HR@5":    avg["HR@5"],
            "HR@10":   avg["HR@10"],
            "HR@20":   avg["HR@20"],
            "R@3":     avg["R@3"],
            "R@5":     avg["R@5"],
            "R@10":    avg["R@10"],
            "R@20":    avg["R@20"],
            "NDCG@3":  avg["NDCG@3"],
            "NDCG@5":  avg["NDCG@5"],
            "NDCG@10": avg["NDCG@10"],
            "NDCG@20": avg["NDCG@20"],
        })

        print(f"\n=== hop_n = {hop_n} averages ===")
        print(avg[[f"P@{k}"   for k in rec_k_values] +
                  [f"HR@{k}"  for k in rec_k_values] +
                  [f"R@{k}"   for k in rec_k_values] +
                  [f"NDCG@{k}"for k in rec_k_values]].round(4))

    sweep_df = pd.DataFrame(sweep_rows).set_index("hop_n")

    csv_dir = Path(__file__).parent / "csv";  csv_dir.mkdir(exist_ok=True)
    sweep_df.to_csv(csv_dir / "metrics_hop_sweep.csv")

    eval_dir = Path(__file__).parent / "eval"; eval_dir.mkdir(exist_ok=True)

    for rec_k in rec_k_values:
        for metric in ["P", "HR", "R", "NDCG"]:
            col = f"{metric}@{rec_k}"
            plt.figure()
            plt.plot(sweep_df.index, sweep_df[col], marker="o", ms=4)
            plt.title(f"{metric}@{rec_k} vs hop_n")
            plt.xlabel("hop_n")
            plt.ylabel(metric)
            plt.grid(True)
            plt.tight_layout()
            fname = f"{metric.lower()}_at_{rec_k}_vs_hop.png"
            plt.savefig(eval_dir / fname, dpi=200)
            plt.close()

if __name__ == "__main__":
    main()