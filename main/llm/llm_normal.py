from __future__ import annotations
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── 1. load model & tokenizer ────────────────────────────────────────
_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
tok = AutoTokenizer.from_pretrained(_MODEL_ID)
mdl = AutoModelForCausalLM.from_pretrained(
    _MODEL_ID,
    device_map="auto",
    torch_dtype="auto"
)
mdl.eval()

# ── 2. scoring helper ────────────────────────────────────────────────
def score_coherence(paragraph: str, abstract: str) -> float:
    """
    Returns a log-likelihood–based coherence score for
    paragraph → abstract. Higher means more coherent.
    """
    # join with a sentinel or separator token if you like
    text = paragraph.strip() + tok.eos_token + abstract.strip() + tok.eos_token
    inputs = tok(text, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        # labels=input_ids so loss is the CE over all tokens
        outputs = mdl(**inputs, labels=inputs.input_ids)
    # huggingface returns mean over tokens; multiply by length to get total log-prob
    total_loss = outputs.loss.item() * inputs.input_ids.size(1)
    # use negative loss as a score (higher = more likely)
    return -total_loss

# ── 3. rerank function ───────────────────────────────────────────────
def llm_loglikelihood_rerank(
    paragraph: str,
    candidates: pd.DataFrame,
    k: int = 10,
    max_candidates: int = 100,
) -> pd.DataFrame:
    """
    Scores each candidate abstract by how well the LLM 'explains' it
    after the paragraph, then returns top‐k most coherent.
    """
    subs = candidates.iloc[:max_candidates].copy()
    scores = []
    for _, row in subs.iterrows():
        coh = score_coherence(paragraph, row.abstract or "")
        scores.append({"pid": row.pid, "coherence_score": coh})
    df_scores = pd.DataFrame(scores)
    ranked = subs.merge(df_scores, on="pid")
    # sort descending: highest (least loss) first
    ranked = ranked.sort_values("coherence_score", ascending=False)
    return ranked.head(k)
