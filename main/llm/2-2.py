
# This is rrf reranking -> llm reranking 
from __future__ import annotations
import faulthandler
faulthandler.enable(all_threads=True, file=open("fault.log", "w"))

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from neo4j import GraphDatabase
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
from ai import sliding_score, RerankError  # returns DataFrame with pid, score
import matplotlib.pyplot as plt         


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
TESTSET_PATH  = Path(os.getenv("TESTSET_PATH", "/home/abhi/Desktop/Manami/recommender-system/datasets/testset2.jsonl"))
MAX_CASES     = int(os.getenv("MAX_CASES", 50)) # Number of test cases to evaluate
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", 0.95))
TOPK_LIST     = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20) # K-values for evaluation metrics

# Neo4j connection
_NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
_NEO4J_PASS = os.getenv("NEO4J_PASS", "secret")
_driver     = GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASS))

# Tokenizer for chunking
TOKENIZER  = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", use_fast=True)

# cosine similarity 
def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T

def evaluate_case(
    paragraph: str,
    true_pids: List[str],
    target_year: Optional[int] = None
) -> dict:
    # take the raw paragraph, clean it (lower-casing, removing punctuation/stopwords, etc.), 
    # so that BM25 and embeddings both work on normalized text.
    cleaned = clean_text(paragraph)
    chunks  = chunk_tokens(cleaned, TOKENIZER, win=256, stride=64)

    # recall & fuse with reciprocal rank fusion (RRF)
    # full-text & vector for whole paragraph
    full_bm25 = recall_fulltext(cleaned).assign(src="full_bm25")
    full_vec  = recall_vector(embed(cleaned)).assign(src="full_vec")
    
    # chunk-level retrieval
    chunk_pool = recall_by_chunks(chunks)

    # apply RRF fusion across all sources
    # full-doc scores (bm25_score, semantic_score) → hybrid_full
    # chunk-level max scores → hybrid_chunk
    # Total hybrid_score = hybrid_full + hybrid_chunk
    # strip out any extraneous columns so rrf_fuse sees just pid & rank
    # strip everything except pid & rank on the two full-doc tables
    for df in (full_bm25, full_vec):
        df.drop(columns=[c for c in df.columns if c not in ("pid","rank")],
                inplace=True)
        
    pool = rrf_fuse(
        full_bm25,
        full_vec,
        chunk_pool,
        top_k=40,
    ).reset_index(drop=True)  

    pool["rrf_rank"] = np.arange(len(pool))

    # and extract pids for reranking
    cand = fetch_metadata(pool["pid"].tolist())[["pid", "title", "abstract", "year"]]
    if target_year is not None:
        cand = cand[cand["year"] < target_year]
    if cand.empty:
        raise ValueError("Empty candidate set after filtering by year & abstract")

    # 4. rerank via LLM scoring
    try:
        llm_df = sliding_score(paragraph,            # ← returns pid, score
                              cand[["pid", "title", "abstract"]],
                              window_size=5,
                              stride=1)    # keep up to 20

        # bring RRF metrics in for tie-breaking
        llm_df = (
            llm_df
            .merge(pool[["pid", "rrf_rank"]], on="pid", how="left")
        )

        # sort: 1) LLM score ↓  2) rrf_rank ↑
        llm_df = llm_df.sort_values(
            by=["score", "rrf_rank"],
            ascending=[False, True],
            kind="mergesort"
        )
        # print(llm_df[["pid", "score", "rrf_rank"]].to_string(index=False))

        predicted = llm_df["pid"].tolist()

    except RerankError as e:
        print("⚠️  Rerank failed, using RRF order:", e)
        # pool is already sorted by rrf_score desc inside rrf_fuse
        predicted = pool["pid"].tolist()

    # embed references & predicted candidates
    # Calculates cosine similarities between reranked abstracts and true references.
    # For each predicted paper, checks if any unmatched reference has sim ≥ 0.95 → it's a hit.
    ref_meta = fetch_metadata(true_pids).dropna(subset=["abstract"])
    ref_ids  = ref_meta["pid"].tolist()
    ref_emb  = np.stack([embed(t) for t in ref_meta["abstract"]]) if ref_ids else np.zeros((0, 768))

    cand_meta = cand.set_index("pid").loc[predicted]
    cand_emb  = np.stack([embed(t) for t in cand_meta["abstract"]])

    # 6. greedy match for hit detection
    sims = cosine_matrix(cand_emb, ref_emb) if ref_ids else np.zeros((len(predicted),0))
    unmatched = set(range(len(ref_ids)))
    hits = []
    for i in range(len(predicted)):
        if not unmatched:
            hits.append(False)
            continue
        idxs = list(unmatched)
        best_j = sims[i, idxs].argmax()
        if sims[i, idxs][best_j] >= SIM_THRESHOLD:
            hits.append(True)
            unmatched.remove(idxs[best_j])
        else:
            hits.append(False)

    # https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems    
    # P@k  = # relevant hits in top-k / k
    # HR@k = # whether at least one hit in top-k
    # R@k  = # hits / total ground-truth
    # NDCG = # relevance-weighted discounted ranking quality
    n_rel = len(ref_ids)
    out   = {}
    for k in TOPK_LIST:
        topk = hits[:k]
        p_at_k = sum(topk) / k
        hr_at_k = float(any(topk))
        r_at_k = sum(topk) / n_rel if n_rel else 0.0
        dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(topk))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(n_rel, k)))
        ndcg = (dcg / idcg) if idcg else 0.0
        out.update({f"P@{k}": p_at_k, f"HR@{k}": hr_at_k, f"R@{k}": r_at_k, f"NDCG@{k}": ndcg})
    return out

# main function
# Runs evaluation over the test cases and prints the average metrics.
# main
def main() -> None:
    df = pd.read_json(TESTSET_PATH, lines=True).head(MAX_CASES)
    rows: list[dict] = []

    # Single pass: evaluate each paragraph once
    for rec in df.to_dict("records"):
        m = evaluate_case(
            rec["paragraph"],
            [str(x) for x in rec.get("references", [])],
            rec.get("year")
        )
        m["method"] = "rrf_llm_working2"     # or "bm25_full" as you prefer
        rows.append(m)

    # Build the DataFrame once
    metric_df = pd.DataFrame(rows)

    # 1) Print averages
    print("\nRRF + LLM (k=20) average metrics:\n")
    print(metric_df.mean(numeric_only=True).round(4))

    # 2) Plot
    ks = np.array(TOPK_LIST)
    for prefix in ["P","HR","R","NDCG"]:
        y = metric_df[[f"{prefix}@{k}" for k in ks]].mean().values
        plt.figure()
        plt.plot(ks, y, marker="o")
        plt.title(f"{prefix}@k vs k")
        plt.xlabel("k")
        plt.ylabel(prefix)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(Path(__file__).parent / "eval" / f"{prefix.lower()}_rrf_llm_working2.png", dpi=200)
        plt.close()

    # 3) Save CSV
    out_path = Path(__file__).parent / "csv2" / "2metrics_rrf_llm_working2.csv"
    out_path.parent.mkdir(exist_ok=True)
    metric_df.to_csv(out_path, index=False)

if __name__ == "__main__":
    try:
        main()
        send_email("✅ Script completed", "Your reranking script finished successfully. 2metrics_rrf_llm_working2.csv")
    except Exception as e:
        send_email("❌ Script failed", f"Your reranking script failed with error:\n\n{e} 2metrics_rrf_llm_working2.csv")
        raise  # re-raise the error for visibility

