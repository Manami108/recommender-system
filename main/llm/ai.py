# This code is to rerank those candidates using LLM based on how relevant each paper is to a given input paragraph. 

from __future__ import annotations
import json, logging, os, re, sys
from pathlib import Path
from typing import List
from collections import defaultdict
import torch
torch.cuda.empty_cache()
import pandas as pd

# disable SDPA integration in Transformers
#  disable structured dot-product attention in PyTorch
import os
os.environ["TRANSFORMERS_NO_ADA"] = "1"
os.environ["TRANSFORMERS_NO_SDP"] = "1"
torch.backends.cuda.sdp_enabled = False      # disable structured dot-product attention in PyTorch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig

# config
_MODEL_ID  = os.getenv("LLAMA_MODEL",  "meta-llama/Meta-Llama-3.1-8B-Instruct")

# path to prompt
# https://www.promptingguide.ai/jp/techniques/cot
# https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/
# https://medium.com/@tahirbalarabe2/prompt-engineering-with-llama-3-3-032daa5999f7
# https://www.kaggle.com/code/manojsrivatsav/prompt-engineering-with-llama-3-1-8b
# https://dev.to/simplr_sh/llm-re-ranking-enhancing-search-and-retrieval-with-ai-28b7
# https://blog.reachsumit.com/posts/2023/12/prompting-llm-for-ranking/#fn:8

_PROMPT_PATH = Path(__file__).parent / "prompts" / "working2.prompt"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# QLoRa
# https://reinforz.co.jp/bizmedia/13036/
# https://note.com/npaka/n/na506c63b8cc9
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,             
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# LLM model 
_tok = AutoTokenizer.from_pretrained(_MODEL_ID, use_default_system_prompt=False)
_mdl = AutoModelForCausalLM.from_pretrained(
    _MODEL_ID,
    device_map="auto",             
    quantization_config=bnb_cfg,  # Using QLoRa for making system lightweight
    trust_remote_code=True          # avoids class-mismatch errors
)

MAX_GEN    = 1700 # max tokens to generate per prompt
BATCH_SIZE = 5 # how many candidates per LLM call
MAX_POOL   = 200  # cap on total candidates before batching
TOK_HEAD = _mdl.config.max_position_embeddings - MAX_GEN   # max context tokens (after tokenization)

_gen = pipeline(
    "text-generation",
    model=_mdl,
    tokenizer=_tok,
    max_new_tokens=MAX_GEN,
    do_sample=False,
    pad_token_id=_tok.eos_token_id,
    return_full_text=False,
)

# load prompt once
_PROMPT_TMPL = _PROMPT_PATH.read_text(encoding="utf-8")

# accept both <RESULT>…</RESULT> and <|RESULT|>…<|/RESULT|>
_JSON_RE = re.compile(
    r"(?:<\|?/?RESULT\|?>)?\s*(\[[\s\S]*?\])\s*(?:<\|?/?RESULT\|?>)?",
    re.MULTILINE,
)
# print(_mdl.config.max_position_embeddings)      # 131072

# this is for any reranking specific failures
class RerankError(RuntimeError):
    pass

# candidates: a DataFrame with columns pid, title, abstract
def score_window(paragraph: str, candidates: pd.DataFrame) -> pd.DataFrame:

    def wrap_prompt(body: str) -> str:
        # wrap with begin_of_text and end_of_turn
        return "<|begin_of_text|>\n" + body.strip() + "\n<|eot_id|>"
    all_scores: list[pd.DataFrame] = []
    for i in range(0, len(candidates), BATCH_SIZE):
        part = candidates.iloc[i : i + BATCH_SIZE]
        cand_block = "\\n\\n".join(
            f"PID: {r.pid}\\nTitle: {r.title}\\nAbstract: {r.abstract}"
            for _, r in part.iterrows()
        )
        sys_prompt = (
            _PROMPT_TMPL
            .replace("<<<PARAGRAPH>>>", paragraph.strip())
            .replace("<<<CANDIDATES>>>", cand_block)
        )
        prompt = wrap_prompt(sys_prompt)
        if len(_tok(prompt).input_ids) > TOK_HEAD:
            raise RerankError("Prompt too long; reduce BATCH_SIZE or truncate abstracts")
        raw_out = _gen(prompt)[0]["generated_text"]
        m = _JSON_RE.search(raw_out)
        if not m:
            raise RerankError("Failed to extract JSON from LLM response")
        tidy = re.sub(r",\\s*(?=[\\]\\}])", "", m.group(1))
        tidy = re.sub(r'"pid"\\s*:\\s*([0-9]+)', r'"pid":"\\1"', tidy)
        rec = json.loads(tidy)
        df_part = pd.DataFrame(rec)
        if "pid" not in df_part.columns and "id" in df_part.columns:
            df_part = df_part.rename(columns={"id": "pid"})
        df_part["pid"] = df_part["pid"].astype(str)
        df_part["score"] = pd.to_numeric(df_part["score"], errors="coerce").fillna(0.0)
        all_scores.append(df_part[["pid", "score"]])
    return pd.concat(all_scores, ignore_index=True)

def sliding_score(
    paragraph: str,
    all_cands: pd.DataFrame,
    window_size: int = 5,
    stride: int = 2,
) -> pd.DataFrame:
    scores_accum = defaultdict(list)

    # Slide windows
    for start in range(0, len(all_cands) - window_size + 1, stride):
        window = all_cands.iloc[start : start + window_size]
        win_scores = score_window(paragraph, window)
        for _, row in win_scores.iterrows():
            scores_accum[row.pid].append(row.score)

    # Handle any candidates not covered by full windows
    covered = set(scores_accum.keys())
    missing = set(all_cands.pid.astype(str)) - covered
    if missing:
        # Score first window slice for missing
        fallback = score_window(paragraph, all_cands.iloc[:window_size])
        for _, row in fallback.iterrows():
            if row.pid in missing:
                scores_accum[row.pid].append(row.score)

    # Compute averages
    avg_list = []
    for pid, vals in scores_accum.items():
        avg = sum(vals) / len(vals)
        avg_list.append({"pid": pid, "score": round(avg, 4)})

    avg_df = pd.DataFrame(avg_list)
    result = (
        all_cands
        .merge(avg_df, on="pid")
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python llama_rerank.py paragraph.txt candidates.tsv")
        sys.exit(1)

    paragraph = Path(sys.argv[1]).read_text(encoding="utf-8")
    cand_df = pd.read_csv(sys.argv[2], sep="\\t", names=["pid", "title", "abstract"])

    try:
        ranked = sliding_score(paragraph, cand_df, window_size=5, stride=1)
        print(ranked.head(10)[["pid", "score"]])
    except RerankError as err:
        logging.error("Rerank failed: %s", err)