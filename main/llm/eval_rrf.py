# This is rrf reranking but no llm 
from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# config
TESTSET_PATH  = Path(os.getenv("TESTSET_PATH",
                     "/home/abhi/Desktop/Manami/recommender-system/datasets/testset_2020_references.jsonl"))
MAX_CASES     = int(os.getenv("MAX_CASES", 100))
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", 0.95))
TOPK_LIST     = tuple(range(1, 21))
TOKENIZER     = AutoTokenizer.from_pretrained(
                    "meta-llama/Meta-Llama-3.1-8B-Instruct", use_fast=True)

# cosine similarity 
def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T

def evaluate_case(
    paragraph: str,
    true_pids: List[str],
    target_year: Optional[int] = None,
) -> dict:
    cleaned = clean_text(paragraph)
    chunks  = chunk_tokens(cleaned, TOKENIZER, win=256, stride=64)

    # 1) Recall
    full_bm25 = recall_fulltext(cleaned).assign(src="full_bm25")
    full_vec  = recall_vector(embed(cleaned)).assign(src="full_vec")
    chunk_pool = recall_by_chunks(chunks)

    # 2) RRF fusion → top-20 list
    for df in (full_bm25, full_vec):
        df.drop(columns=[c for c in df.columns if c not in ("pid", "rank")], inplace=True)

    pool = rrf_fuse(
        full_bm25,
        full_vec,
        chunk_pool,
        top_k=20,
    )                                   # already sorted by RRF desc

    predicted = pool["pid"].tolist()    # *** no LLM rerank ***

    # 3) metadata for evaluation (needed for embeddings)
    cand_meta = fetch_metadata(predicted)[["pid", "abstract", "year"]]
    if target_year is not None:
        cand_meta = cand_meta[cand_meta["year"] < target_year]

    # 4) embed refs & preds
    ref_meta = fetch_metadata(true_pids).dropna(subset=["abstract"])
    ref_ids  = ref_meta["pid"].tolist()
    ref_emb  = np.stack([embed(t) for t in ref_meta["abstract"]]) if ref_ids else np.zeros((0, 768))

    cand_emb = np.stack([embed(t) for t in cand_meta.set_index("pid").loc[predicted]["abstract"]])

    sims       = cosine_matrix(cand_emb, ref_emb) if ref_ids else np.zeros((len(predicted), 0))
    unmatched  = set(range(len(ref_ids)))
    hits: list[bool] = []

    for i in range(len(predicted)):
        if not unmatched:
            hits.append(False)
            continue
        idxs = list(unmatched)
        best = sims[i, idxs].argmax()
        if sims[i, idxs][best] >= SIM_THRESHOLD:
            hits.append(True)
            unmatched.remove(idxs[best])
        else:
            hits.append(False)

    # 5) metrics
    n_rel = len(ref_ids)
    out   = {}
    for k in TOPK_LIST:
        topk  = hits[:k]
        p     = sum(topk) / k
        hr    = float(any(topk))
        r     = sum(topk) / n_rel if n_rel else 0.0
        dcg   = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(topk))
        idcg  = sum(1 / np.log2(i + 2) for i in range(min(n_rel, k)))
        ndcg  = (dcg / idcg) if idcg else 0.0
        out.update({f"P@{k}": p, f"HR@{k}": hr, f"R@{k}": r, f"NDCG@{k}": ndcg})

    return out

# main
def main() -> None:
    if not TESTSET_PATH.exists():
        raise FileNotFoundError(f"Testset not found at {TESTSET_PATH}")

    tests = pd.read_json(TESTSET_PATH, lines=True).head(MAX_CASES)

    rows: list[dict] = []
    for rec in tests.to_dict("records"):
        rows.append(
            evaluate_case(
                rec["paragraph"],
                [str(pid) for pid in rec.get("references", [])],
                rec.get("year"),
            )
        )

    metric_df = pd.DataFrame(rows)
    out_path = Path(__file__).parent / "csv" / "metrics_rrf.csv"
    metric_df.to_csv(out_path, index=False)

    avg = metric_df.mean(numeric_only=True).round(4)
    print("\nAverage metrics (pure RRF):\n", avg)

    ks = np.array(TOPK_LIST)
    for prefix in ["P", "HR", "R", "NDCG"]:
        y = metric_df[[f"{prefix}@{k}" for k in ks]].mean().values
        plt.figure()
        plt.plot(ks, y, marker="o")
        plt.title(f"{prefix}@k vs k")
        plt.xlabel("k")
        plt.ylabel(prefix)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(Path(__file__).parent / "eval" / f"{prefix.lower()}_rrf.png", dpi=200)
        plt.close()

if __name__ == "__main__":
    main()
