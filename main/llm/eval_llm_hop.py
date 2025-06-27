
from __future__ import annotations
import os, json, re, math, logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # head-less
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
from transformers import AutoTokenizer

from chunking       import clean_text
from recall         import recall_fulltext, fetch_metadata, embed
from rerank_llm     import rerank_batch, RerankError          
from hop_reasoning  import multi_hop_topic_citation_reasoning 
# ─────────────────────────  GLOBALS & PARAMS  ───────────────────────────────
TESTSET_PATH = Path(os.getenv("TESTSET_PATH",
                    "/home/abhi/Desktop/Manami/recommender-system/datasets/"
                    "testset_2020_references.jsonl"))
MAX_CASES    = int(os.getenv("MAX_CASES", 5))

SEEDS        = 5               # take this many highest-scoring LLM papers as seeds
HOP_TOP_N    = 20              # return ≤ 40 hop papers for each source query
FINAL_K      = 60              # final candidate pool fed to 2nd-stage LLM
BATCH_1      = 5               # batch size stage-1
BATCH_2      = 5              # batch size stage-2
TOPK_LIST    = tuple(range(1, 21))
SIM_THRESH   = 0.95            # cosine threshold to count a “hit”

# ───────────────────────────  HELPERS  ──────────────────────────────────────
_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct",
                                     use_fast=True)
def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T

# ───────────────────────  EVALUATION PER PARAGRAPH  ─────────────────────────
def evaluate_case(paragraph: str,
                  gold_pids: List[str],
                  target_year: Optional[int] = None) -> dict:

    # ----------  STAGE-0  full-text recall  ---------------------------------
    cleaned = clean_text(paragraph)
    bm25    = recall_fulltext(cleaned, k=40)          # top-20 BM25 only
    meta0   = fetch_metadata(bm25.pid.tolist()).dropna(subset=["abstract"])
    pool0   = bm25.merge(meta0, on="pid")
    if target_year is not None:
        pool0 = pool0[pool0.year < target_year]

    # ----------  STAGE-1  LLM rerank  (quick)  -------------------------------
    try:
        rr1 = rerank_batch(paragraph,
                           pool0[["pid","title","abstract"]],
                           k=SEEDS,
                           max_candidates=len(pool0),
                           batch_size=BATCH_1)
        seeds = rr1.pid.tolist()
    except RerankError as e:
        logging.warning("Stage-1 rerank failed → fall back to BM25 order (%s)", e)
        seeds = pool0.pid.head(SEEDS).tolist()

    # ----------  Hop reasoning  ---------------------------------------------
    hops_df = multi_hop_topic_citation_reasoning(seeds,
                                                 top_n=HOP_TOP_N)
    # de-dup & union
    union_ids = list(dict.fromkeys(seeds + hops_df.pid.tolist()))[:FINAL_K]
    meta1     = fetch_metadata(union_ids).dropna(subset=["abstract"])

    # ----------  STAGE-2  LLM rerank (final)  --------------------------------
    try:
        rr2 = rerank_batch(paragraph,
                           meta1[["pid","title","abstract"]],
                           k=FINAL_K,
                           max_candidates=FINAL_K,
                           batch_size=BATCH_2)
        final_ids = rr2.pid.tolist()
    except RerankError as e:
        logging.warning("Stage-2 rerank failed → keep union order (%s)", e)
        final_ids = meta1.pid.tolist()

    # ----------  offline relevance scoring  ---------------------------------
    refs = fetch_metadata(gold_pids).dropna(subset=["abstract"])
    ref_ids  = refs.pid.tolist()
    ref_emb  = np.stack([embed(a) for a in refs.abstract]) if len(refs) else np.zeros((0,768))
    cand_emb = np.stack([embed(a) for a in meta1.set_index("pid").loc[final_ids].abstract])

    sims = cosine_matrix(cand_emb, ref_emb) if len(refs) else np.zeros((len(final_ids),0))
    unmatched = set(range(len(ref_ids)))
    hits = []
    for i in range(len(final_ids)):
        if not unmatched:
            hits.append(False); continue
        idxs = list(unmatched)
        j    = sims[i, idxs].argmax()
        if sims[i, idxs][j] >= SIM_THRESH:
            hits.append(True)
            unmatched.remove(idxs[j])
        else:
            hits.append(False)

    # ----------  metrics  ----------------------------------------------------
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

# ───────────────────────────  MAIN LOOP  ────────────────────────────────────
def main() -> None:

    if not TESTSET_PATH.exists():
        raise FileNotFoundError(TESTSET_PATH)
    df = pd.read_json(TESTSET_PATH, lines=True).head(MAX_CASES)

    metrics = [evaluate_case(rec["paragraph"],
                             [str(x) for x in rec.get("references", [])],
                             rec.get("year"))
               for rec in df.to_dict("records")]

    metric_df = pd.DataFrame(metrics)
    ks = np.array(TOPK_LIST)

        # 3) evaluate each paragraph and tag with this method
    rows: List[dict] = []
    for rec in df.to_dict("records"):
        m = evaluate_case(
            rec["paragraph"],
            [str(pid) for pid in rec.get("references", [])],
            rec.get("year")
        )
        m["method"] = "bm25_hop_rerank"   # tag this run
        rows.append(m)

    # 4) build a DataFrame and dump to CSV
    metric_df = pd.DataFrame(rows)
    out_csv   = Path(__file__).parent / "csv" / "metrics_bm25_hop_rerank.csv"
    metric_df.to_csv(out_csv, index=False)
    print(f"\nSaved per-paragraph metrics to {out_csv}")

    # 5) print average metrics
    avg = metric_df.mean(numeric_only=True).round(4)
    print("\nAverage metrics for BM25 → hop → LLM rerank:\n", avg)

    save_dir = Path(__file__).parent / "eval"
    save_dir.mkdir(exist_ok=True)

    for pref in ["P","HR","R","NDCG"]:
        y = metric_df[[f"{pref}@{k}" for k in ks]].mean().values
        plt.figure()
        plt.plot(ks, y, marker="o")
        plt.title(f"{pref}@k vs k  (avg over {len(metric_df)} paragraphs)")
        plt.xlabel("k (# recommended papers)");   plt.ylabel(pref)
        plt.grid(True); plt.tight_layout()
        plt.savefig(save_dir / f"{pref.lower()}_hop.png", dpi=200)
        plt.close()

    print("\nAverage metrics over "
          f"{len(metric_df)} paragraphs:\n",
          metric_df.mean(numeric_only=True).round(4))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    main()


