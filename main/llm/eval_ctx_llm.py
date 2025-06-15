#!/usr/bin/env python
"""
Offline evaluation:
1. recall papers (BM25 + vectors, full + chunks)
2. rerank with Llama-3 chat prompt (rerank_batch)
3. score against gold references (one-to-one greedy match)
"""

from __future__ import annotations
import os, numpy as np, pandas as pd
from pathlib import Path
from typing import List, Optional
from neo4j import GraphDatabase
from transformers import AutoTokenizer

from chunking import clean_text, chunk_tokens
from recall   import recall_fulltext, recall_vector, fetch_metadata, embed
from rerank2 import rerank_batch, RerankError      # <─ NEW import

# ───────── CONFIG ──────────
TESTSET_PATH  = Path("/home/abhi/Desktop/Manami/recommender-system/datasets/testset_2020_references.jsonl")
MAX_CASES     = 25
SIM_THRESHOLD = 0.95          # a bit looser than before
TOPK_LIST     = (3, 5, 10, 15, 20)

NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "secret")
driver     = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

TOKENIZER  = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# ───────── helpers ─────────
def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T


# ───────── single-case eval ─────────
def evaluate_case(
    paragraph: str,
    true_pids: List[str],
    target_year: Optional[int] = None
) -> dict:

    # 1. chunk paragraph
    cleaned = clean_text(paragraph)
    chunks  = chunk_tokens(cleaned, TOKENIZER, win=128, stride=64)

    # 2. recall pool (4 sources)
    bm25_full  = recall_fulltext(cleaned, k=40).assign(src="full_bm25")
    chunk_bm25 = (
        pd.concat([recall_fulltext(ch, k=40) for ch in chunks])
          .drop_duplicates("pid").assign(src="chunk_bm25")
    )
    full_vec   = recall_vector(embed(cleaned), k=40, sim_threshold=0.30)\
                   .assign(src="full_vec")
    chunk_vec  = (
        pd.concat([recall_vector(embed(ch), k=40, sim_threshold=0.30) for ch in chunks])
          .drop_duplicates("pid").assign(src="chunk_vec")
    )

    pool = pd.concat([bm25_full, chunk_bm25, full_vec, chunk_vec], ignore_index=True)
    pool = pool.sort_values(["src", "sim"], ascending=[True, False])\
               .drop_duplicates("pid")

    # 3. fetch metadata + filter by year
    meta = fetch_metadata(pool["pid"].tolist())
    cand = (pool
            .merge(meta[["pid", "abstract", "year"]], on="pid")
                 .dropna(subset=["abstract"]))
    if target_year is not None:
        cand = cand[cand["year"] <= target_year]
    if cand.empty:
        raise ValueError("Empty candidate set after year filter")

    # 4. Llama-3 rerank
    try:
        reranked = rerank_batch(
            paragraph,
            cand[["pid", "title", "abstract"]],
            k=20,                # top-k to keep
            max_candidates=15,   # truncate for prompt
            batch_size=5
        )
        predicted = reranked["pid"].tolist()
    except RerankError as e:
        print("⚠️  Rerank failed, fallback to original order:", e)
        predicted = cand["pid"].tolist()

    # 5. embed refs & candidates
    ref_meta = fetch_metadata([str(pid) for pid in true_pids]).dropna(subset=["abstract"])
    ref_ids  = ref_meta["pid"].tolist()
    ref_emb  = np.stack([embed(a) for a in ref_meta["abstract"]]) if ref_ids else np.zeros((0,768))

    cand_emb = np.stack([embed(a) for a in
                         cand.set_index("pid").loc[predicted, "abstract"].tolist()])

    # 6. greedy 1-to-1 match
    sims = cosine_matrix(cand_emb, ref_emb) if ref_ids else np.zeros((len(predicted), 0))
    unmatched = set(range(len(ref_ids)))
    hits = []
    for i in range(len(predicted)):
        if not unmatched:
            hits.append(False); continue
        idxs = list(unmatched)
        j    = sims[i, idxs].argmax()
        if sims[i, idxs][j] >= SIM_THRESHOLD:
            hits.append(True); unmatched.remove(idxs[j])
        else:
            hits.append(False)

    # 7. metrics
    n_rel = len(ref_ids)
    out   = {}
    for k in TOPK_LIST:
        topk = hits[:k]
        out[f"P@{k}"]   = sum(topk)/k
        out[f"HR@{k}"]  = float(any(topk))
        out[f"R@{k}"]   = sum(topk)/n_rel if n_rel else 0.0
        dcg  = sum(r/np.log2(i+2) for i, r in enumerate(topk))
        idcg = sum(1/np.log2(i+2) for i in range(min(n_rel, k)))
        out[f"NDCG@{k}"] = dcg/idcg if idcg else 0.0
    return out


# ───────── driver ─────────
def main() -> None:
    if not TESTSET_PATH.exists():
        raise FileNotFoundError(f"Testset not found at {TESTSET_PATH}")
    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        df = pd.read_json(f, lines=True).head(MAX_CASES)
    metrics = [evaluate_case(row["paragraph"], row.get("references", []), row.get("year"))
               for row in df.to_dict("records")]
    avg = pd.DataFrame(metrics).mean(numeric_only=True)
    print("\nLLM rerank - average metrics:\n", avg.round(4))


if __name__ == "__main__":
    main()
