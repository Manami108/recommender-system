# eval_rrf_llm_hop.py  (20 RRF seeds → hop → LLM top-20)

from __future__ import annotations
import faulthandler
faulthandler.enable(all_threads=True, file=open("fault.log", "w"))

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
from rerank_llm import sliding_score, RerankError
from hop_reasoning import multi_hop_topic_citation_reasoning


import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(subject: str, body: str):
    sender_email = "manakokko25@gmail.com"
    receiver_email = "manakokko25@gmail.com"
    password = "ehjijtgqzwdahiay"
    
    # Gmail: use 'smtp.gmail.com', port 587
    # Outlook: 'smtp.office365.com', port 587
    # Others: Check your provider's SMTP server & port
    smtp_server = "smtp.gmail.com"
    port = 587

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(smtp_server, port)
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        print("✅ Email notification sent.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")



# config
TESTSET_PATH = Path(os.getenv(
    "TESTSET_PATH",
    "/home/abhi/Desktop/Manami/recommender-system/datasets/testset1.jsonl"))
MAX_CASES  = int(os.getenv("MAX_CASES", 50))
TOPK_LIST  = tuple(range(1, 21))
SIM_THRESH = 0.95

RRF_TOPK       = 40  # keep top 20 seeds after RRF fusion
HOP_TOP_N      = 2   # retrieve up to 20 hop papers per seed
FINAL_POOL_CAP = 120  # cap total pool size before final LLM
LLM_TOPK       = max(TOPK_LIST)  # final list size

TOKENIZER = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct", use_fast=True)

# cosin similarity 
def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T

# 
def evaluate_case(
    paragraph: str,
    gold_pids: List[str],
    target_year: Optional[int] = None,
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

    seeds_df = (
        rrf_fuse(bm25, vec, chunk_pool, top_k=RRF_TOPK)
        .reset_index(drop=True)
    )
    # add a deterministic rank (0 = best, 1 = next…)
    seeds_df["rrf_rank"] = np.arange(len(seeds_df))
    seed_ids = seeds_df.pid.tolist()


    # this is hop expansions
    hop_df   = multi_hop_topic_citation_reasoning(seed_ids, top_n=HOP_TOP_N)  # <<< hop >>>
    union_ids = list(dict.fromkeys(seed_ids + hop_df.pid.tolist()))[:FINAL_POOL_CAP]  # <<< hop >>>
    raw_meta = fetch_metadata(union_ids).dropna(subset=["abstract"])
    if target_year is not None:
        raw_meta = raw_meta[raw_meta.year < target_year]

    # re‐order to match `union_ids`
    meta_pool = (
        raw_meta.set_index("pid")
                .loc[union_ids]           # select & reorder
                .reset_index()            # bring pid back as a column
    )

    if target_year is not None:
        meta_pool = meta_pool[meta_pool.year < target_year]

    # LLM-based reranking via sliding_score
    rerank_failed = False
    try:
        llm_df = sliding_score(
            paragraph,
            meta_pool[["pid", "title", "abstract"]]
        )
        # keep top LLM_TOPK
        llm_df = llm_df.nlargest(LLM_TOPK, "score")
        # merge RRF rank for tie-breaking
        llm_df = llm_df.merge(
            seeds_df[["pid", "rrf_rank"]], on="pid", how="left"
        )
        llm_df = llm_df.sort_values(
            ["score", "rrf_rank"], ascending=[False, True], kind="mergesort"
        )
        final_ids = llm_df.pid.tolist()
    except RerankError as e:
        logging.warning("⚠️ sliding_score failed, falling back to RRF seeds: %s", e)
        final_ids = seeds_df.pid.tolist()

    # embed references & predicted candidates
    # Calculates cosine similarities between reranked abstracts and true references.
    # For each predicted paper, checks if any unmatched reference has sim ≥ 0.95 → it's a hit.
    refs = fetch_metadata(gold_pids).dropna(subset=["abstract"])
    ref_ids = refs.pid.tolist()
    ref_emb = np.stack([embed(a) for a in refs.abstract]) if ref_ids else np.zeros((0,768))
    cand_abstracts = meta_pool.set_index("pid").loc[final_ids].abstract
    cand_emb = np.stack([embed(a) for a in cand_abstracts])

    # greedy match for hit detection
    sims = cosine_matrix(cand_emb, ref_emb) if ref_ids else np.zeros((len(final_ids),0))
    unmatched, hits = set(range(len(ref_ids))), []
    for i in range(len(final_ids)):
        if not unmatched:
            hits.append(False)
            continue
        idxs = list(unmatched)
        j = sims[i, idxs].argmax()
        if sims[i, idxs][j] >= SIM_THRESH:
            hits.append(True)
            unmatched.remove(idxs[j])
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
    out["method"]        = "rrf_hop2_llm"
    out["rerank_failed"] = rerank_failed
    return out

import matplotlib
matplotlib.use("Agg")          # off-screen backend

# main
def main() -> None:
    df = pd.read_json(TESTSET_PATH, lines=True).head(MAX_CASES)
    rows: list[dict] = []
    for rec in df.to_dict("records"):
        m = evaluate_case(
            rec["paragraph"],
            [str(pid) for pid in rec.get("references", [])],
            rec.get("year"),
        )
        rows.append(m)

    metric_df = pd.DataFrame(rows)

    # 1) console average
    print("\nRRF → hop → LLM (k=20) average metrics:\n")
    print(metric_df.mean(numeric_only=True).round(4))

    # Plot metrics vs k
    ks = np.array(TOPK_LIST)
    save_dir = Path(__file__).parent / "eval"
    save_dir.mkdir(exist_ok=True)
    for prefix in ["P", "HR", "R", "NDCG"]:
        y = metric_df[[f"{prefix}@{k}" for k in ks]].mean().values
        plt.figure()
        plt.plot(ks, y, marker="o")
        plt.title(f"{prefix}@k vs k")
        plt.xlabel("k")
        plt.ylabel(prefix)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / f"{prefix.lower()}_rrf_hop2_llm.png", dpi=200)
        plt.close()

    # 3) persist CSV
    csv_dir = Path(__file__).parent / "csv1"
    csv_dir.mkdir(exist_ok=True)
    out_path = csv_dir / "1metrics_rrf_hop2_llm.csv"
    metric_df.to_csv(out_path, index=False)

    # Notification
    send_email(
        "✅ eval_rrf_hop2_llm completed",
        f"Your reranking script finished successfully. Metrics saved to {out_path}"
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        send_email(
            "❌ eval_rrf_hop2_llm failed",
            f"Your reranking script failed with error:\n\n{e}"
        )
        raise


