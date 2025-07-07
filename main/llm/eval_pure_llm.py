# rrf_pure_llama_reranking.py
"""
Pure-LLaMA reranking evaluation script.
Reads JSONL test set, performs RRF fusion, reranks with LLaMA, computes metrics, plots, emails.
"""
from __future__ import annotations
import faulthandler
faulthandler.enable(all_threads=True, file=open("fault.log", "w"))

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from chunking import clean_text, chunk_tokens
from recall import (
    recall_fulltext,
    recall_vector,
    recall_by_chunks,
    rrf_fuse,
    fetch_metadata,
    embed,
)

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


# Config
TESTSET_PATH  = Path(os.getenv("TESTSET_PATH", "/home/abhi/Desktop/Manami/recommender-system/datasets/testset1.jsonl"))
if not TESTSET_PATH.exists():
    raise FileNotFoundError(f"Test set not found: {TESTSET_PATH}")

MAX_CASES     = int(os.getenv("MAX_CASES", 50))
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", 0.95))
TOPK_LIST     = tuple(range(1, 21))

# LLaMA model & tokenizer
MODEL_NAME = os.getenv("LLAMA_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
TOKENIZER  = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
LLAMA      = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).cuda()
LLAMA.eval()


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T


def llama_rate(paragraph: str, title: str, abstract: str) -> int:
    prompt = (
        f"Given the research paragraph:\n'''{paragraph}'''\n"
        f"Paper Title: {title}\nAbstract: {abstract}\n"
        "Rate relevance 1 (not relevant) to 5 (highly relevant). Respond with a single digit."
    )
    inputs = TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    with torch.no_grad():
        out = LLAMA.generate(
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


def evaluate_case(paragraph: str, true_pids: List[str], target_year: Optional[int]) -> dict:
    # 1. Clean and chunk
    clean = clean_text(paragraph)
    chunks = chunk_tokens(clean, TOKENIZER, win=256, stride=64)

    # 2. Retrieval + RRF
    bm25 = recall_fulltext(clean).assign(src="bm25")
    vec  = recall_vector(embed(clean)).assign(src="vec")
    chunk_pool = recall_by_chunks(chunks)
    for df in (bm25, vec):
        df.drop(columns=[c for c in df if c not in ("pid","rank")], inplace=True)
    pool = rrf_fuse(bm25, vec, chunk_pool, top_k=40).reset_index(drop=True)
    pool["rrf_rank"] = np.arange(len(pool))

    # 3. Metadata
    cand = fetch_metadata(pool["pid"].tolist())[ ["pid","title","abstract","year"] ]
    if target_year:
        cand = cand[cand["year"] < target_year]
    if cand.empty:
        raise ValueError("No candidates after year filter.")

    # 4. Pure LLaMA rerank
    rates = []
    for _, row in cand.iterrows():
        rates.append((row["pid"], llama_rate(paragraph, row["title"], row["abstract"])))
    llm_df = pd.DataFrame(rates, columns=["pid","score"]).merge(
        pool[["pid","rrf_rank"]], on="pid"
    )
    llm_df.sort_values(by=["score","rrf_rank"], ascending=[False,True], inplace=True)
    predicted = llm_df["pid"].tolist()

    # 5. Hit detection
    ref_meta = fetch_metadata(true_pids).dropna(subset=["abstract"])
    ref_ids  = ref_meta["pid"].tolist()
    ref_emb  = np.stack([embed(a) for a in ref_meta["abstract"]]) if ref_ids else np.zeros((0,768))
    cand_emb = np.stack([embed(fetch_metadata([pid])["abstract"].iloc[0]) for pid in predicted])
    sims     = cosine_matrix(cand_emb, ref_emb) if ref_ids else np.zeros((len(predicted),0))
    unmatched = set(range(len(ref_ids)))
    hits = []
    for i in range(len(predicted)):
        if not unmatched:
            hits.append(False)
        else:
            idxs = list(unmatched)
            j = sims[i, idxs].argmax()
            if sims[i, idxs][j] >= SIM_THRESHOLD:
                hits.append(True)
                unmatched.remove(idxs[j])
            else:
                hits.append(False)

    # 6. Metrics
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


def main():
    df = pd.read_json(TESTSET_PATH, lines=True).head(MAX_CASES)
    rows = []
    for rec in df.to_dict('records'):
        m = evaluate_case(rec['paragraph'], [str(x) for x in rec.get('references',[])], rec.get('year'))
        m['method'] = 'rrf_pure_llama'
        rows.append(m)

    metric_df = pd.DataFrame(rows)
    print("\nAverage metrics:\n", metric_df.mean(numeric_only=True).round(4))

    ks = np.array(TOPK_LIST)
    for p in ["P","HR","R","NDCG"]:
        vals = metric_df[[f"{p}@{k}" for k in ks]].mean().values
        plt.figure(); plt.plot(ks, vals, marker='o')
        plt.xlabel('k'); plt.ylabel(p); plt.title(f"{p}@k vs k"); plt.grid(True)
        png = Path(__file__).parent / 'eval' / f'{p.lower()}_rrf_pure_llama.png'
        png.parent.mkdir(exist_ok=True)
        plt.savefig(png, dpi=200); plt.close()

    out_csv = Path(__file__).parent / 'csv1' / '1metrics_rrf_pure_llama.csv'
    out_csv.parent.mkdir(exist_ok=True)
    metric_df.to_csv(out_csv, index=False)
    send_email("✅ Completed", f"Metrics saved to {out_csv}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        send_email("❌ Failed", f"Error: {e}")
        raise
