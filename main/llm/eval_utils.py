# ──────────────────────────────────────────────────────────────
# evaluation_v2.py               (save alongside your project)
# ──────────────────────────────────────────────────────────────
from __future__ import annotations
import os, json, random, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional
from neo4j import GraphDatabase
from transformers import AutoTokenizer

from chunking    import clean_text, chunk_tokens
from recall      import (recall_fulltext, recall_vector, recall_by_chunks,
                         rrf_fuse, fetch_metadata, embed)
from rerank_llm  import rerank_batch, RerankError

# ----------------------------------------------------------------
# reproducibility: disable CuDNN autotune & set seeds
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
random.seed(0); np.random.seed(0); torch.manual_seed(0)
# ----------------------------------------------------------------

TESTSET_PATH = Path(
    os.getenv("TESTSET_PATH",
              "/home/abhi/Desktop/Manami/recommender-system/datasets/testset_2020_references.jsonl"))
MAX_CASES     = int(os.getenv("MAX_CASES", 25))   # evaluate this many rows
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", 0.95))
TOPK_LIST     = (3, 5, 10, 15, 20)

# Neo4j
_NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
_NEO4J_PASS = os.getenv("NEO4J_PASS", "secret")
_driver     = GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASS))

# Tokeniser for chunking
TOKENIZER = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct", use_fast=True
)

# -------------- helper --------------------------------------------------------
def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T

# -------------- per-paragraph evaluation -------------------------------------
def evaluate_case(paragraph: str,
                  true_pids: List[str],
                  target_year: Optional[int] = None) -> dict:
    cleaned = clean_text(paragraph)
    chunks  = chunk_tokens(cleaned, TOKENIZER, win=128, stride=64)

    full_bm25 = recall_fulltext(cleaned).assign(src="full_bm25")
    full_vec  = recall_vector(embed(cleaned)).assign(src="full_vec")
    chunk_pool = recall_by_chunks(chunks)

    for df in (full_bm25, full_vec):
        df.drop(columns=[c for c in df.columns if c not in ("pid", "rank")],
                inplace=True)

    pool = rrf_fuse(full_bm25, full_vec, chunk_pool, k_rrf=60, top_k=60)
    cand_ids = pool["pid"].head(60).tolist()
    meta     = fetch_metadata(cand_ids).dropna(subset=["abstract"])
    if target_year is not None:
        meta = meta[meta["year"] < target_year]
    if meta.empty:
        raise ValueError("No candidates after filtering")

    try:
        reranked = rerank_batch(
            paragraph, meta[["pid", "title", "abstract"]],
            k=60, max_candidates=60, batch_size=3
        )
        ranked_ids = reranked["pid"].tolist()
    except RerankError as e:
        print("⚠️  LLM failed → fallback:", e)
        ranked_ids = meta["pid"].tolist()

    # embed references + candidates
    ref_meta = fetch_metadata(true_pids).dropna(subset=["abstract"])
    ref_ids  = ref_meta["pid"].tolist()
    ref_emb  = (np.stack([embed(t) for t in ref_meta["abstract"]])
                if ref_ids else np.zeros((0, 768)))
    cand_meta = meta.set_index("pid").loc[ ranked_ids ]
    cand_emb  = np.stack([embed(t) for t in cand_meta["abstract"]])

    sims = (cosine_matrix(cand_emb, ref_emb)
            if ref_ids else np.zeros((len(ranked_ids), 0)))
    unmatched = set(range(len(ref_ids)))
    hits = []
    for i in range(len(ranked_ids)):
        if not unmatched:
            hits.append(False); continue
        idxs = list(unmatched)
        best = sims[i, idxs].argmax()
        if sims[i, idxs][best] >= SIM_THRESHOLD:
            hits.append(True); unmatched.remove(idxs[best])
        else:
            hits.append(False)

    n_rel, out = len(ref_ids), {}
    for k in TOPK_LIST:
        topk = hits[:k]
        dcg  = sum(r / np.log2(i+2) for i, r in enumerate(topk))
        idcg = sum(1 / np.log2(i+2) for i in range(min(n_rel, k)))
        out.update({
            f"P@{k}"    : sum(topk) / k,
            f"HR@{k}"   : float(any(topk)),
            f"R@{k}"    : sum(topk) / n_rel if n_rel else 0.0,
            f"NDCG@{k}" : dcg / idcg if idcg else 0.0,
        })
    return out
# ------------------------------------------------------------------------------

def main():
    if not TESTSET_PATH.exists():
        raise FileNotFoundError(TESTSET_PATH)
    df  = pd.read_json(TESTSET_PATH, lines=True).head(MAX_CASES)
    res = [evaluate_case(rec["paragraph"],
                         [str(pid) for pid in rec.get("references", [])],
                         rec.get("year"))
           for rec in df.to_dict("records")]
    metrics = pd.DataFrame(res)

    # ► NEW ─── draw one figure per metric ─────────────
    for metric_prefix in ["P", "HR", "R", "NDCG"]:
        ks   = np.array(TOPK_LIST)
        vals = metrics[[f"{metric_prefix}@{k}" for k in TOPK_LIST]].mean().values
        plt.figure()
        plt.plot(ks, vals, marker="o")
        plt.title(f"{metric_prefix}@k vs k  (avg of {len(metrics)} cases)")
        plt.xlabel("k (# recommended papers)")
        plt.ylabel(metric_prefix)
        plt.grid(True)
        plt.tight_layout()
        plt.show()                      # one window per metric
    # ──────────────────────────────────────────────────

    print("\nAverage metrics across",
          len(metrics), "cases\n", metrics.mean().round(4))

if __name__ == "__main__":
    main()
