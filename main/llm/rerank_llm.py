# This code is to rerank those candidates using LLM based on how relevant each paper is to a given input paragraph. 

from __future__ import annotations
import json, logging, os, re, sys
from pathlib import Path
from typing import List

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# config
_MODEL_ID  = os.getenv("LLAMA_MODEL",  "meta-llama/Meta-Llama-3-8B-Instruct")
_DEVICE    = os.getenv("LLAMA_DEVICE", "auto")
MAX_GEN    = 8192 # max tokens to generate per prompt
MAX_ABS_CH = 750  # max characters of abstract to include
BATCH_SIZE = 3 # how many candidates per LLM call
MAX_POOL   = 60  # cap on total candidates before batching
TOK_HEAD   = 6144  # max context tokens (after tokenization)

# path to prompt
# https://www.promptingguide.ai/jp/techniques/cot
# https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/
# https://medium.com/@tahirbalarabe2/prompt-engineering-with-llama-3-3-032daa5999f7


_PROMPT_PATH = Path(__file__).parent / "prompts" / "cars2.prompt"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# LLM model 
_tok = AutoTokenizer.from_pretrained(_MODEL_ID, use_default_system_prompt=False)
_mdl = AutoModelForCausalLM.from_pretrained(_MODEL_ID, device_map=_DEVICE, torch_dtype="auto")
_gen = pipeline(
    "text-generation",
    model=_mdl,
    tokenizer=_tok,
    max_new_tokens=MAX_GEN,
    do_sample=False,
    # temperature=0,
    # top_p=1,
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

# this is for any reranking specific failures
class RerankError(RuntimeError):
    pass

# candidates: a DataFrame with columns pid, title, abstract
def rerank_batch(
    paragraph: str,
    candidates: pd.DataFrame,
    *,
    k: int = 20, # This is how many final top papers to return 
    max_candidates: int = MAX_POOL, # drop any beyond this before batching.
    batch_size: int = BATCH_SIZE, # how many candidates per LLM API call.
) -> pd.DataFrame:
    if len(candidates) > max_candidates:
        candidates = candidates.iloc[:max_candidates].copy()

    def wrap_prompt(body: str) -> str:
        # wrap with begin_of_text and end_of_turn
        return (
            "<|begin_of_text|>\n"
            f"{body.strip()}\n"
            "<|eot_id|>"
        )
    # Loops over the candidates in chunks of batch_size (e.g. 3 at a time).
    all_scores: List[pd.DataFrame] = []
    for start in range(0, len(candidates), batch_size):
        part = candidates.iloc[start : start + batch_size]

        # build candidate block
        def trunc(t: str) -> str:
            return t if len(t) <= MAX_ABS_CH else t[:MAX_ABS_CH] + " …"
        
        # Concatenates each candidate’s ID, title, and a truncated abstract (so you don’t blow past the token limit).
        cand_block = "\n\n".join(
            f"PID: {r.pid}\nTitle: {r.title}\nAbstract: {trunc(r.abstract)}"
            for _, r in part.iterrows()
        )

        # fill template placeholders
        sys_prompt = (
            _PROMPT_TMPL
            .replace("<<<PARAGRAPH>>>", paragraph.strip())
            .replace("<<<CANDIDATES>>>", cand_block)
        )
        prompt = wrap_prompt(sys_prompt)

        # Ensure context length
        if len(_tok(prompt).input_ids) > TOK_HEAD:
            raise RerankError("Prompt too long; reduce batch size or truncate abstracts")

        # Generate
        raw_out = _gen(prompt)[0]["generated_text"]
        raw = re.sub(r"<\|(?:eot_id|eom_id)\|>.*$", "", raw_out, flags=re.DOTALL).strip()

        # Extract JSON object containing pid + score
        m = _JSON_RE.search(raw)
        if not m:
            raise RerankError(f"No JSON object found in LLM output:\n{raw[:300]}")
        json_text = m.group(1)

        # Fix pid quoting
        json_text = re.sub(r'"pid"\s*:\s*([0-9]+)', r'"pid":"\1"', json_text)

        # Parse
        try:
            rec = json.loads(json_text)
        except Exception as e:
            raise RerankError(f"JSON parse failed: {e}\nSnippet:\n{json_text[:200]}")

        df = pd.DataFrame(rec)
        df.columns = [c.lower() for c in df.columns]
        if "pid" not in df or "score" not in df:
            raise RerankError(f"Expected 'pid' and 'score' fields, got {df.columns.tolist()}")
        df["pid"] = df["pid"].astype(str)
        df["score"] = pd.to_numeric(df["score"], errors="coerce")

        all_scores.append(df[["pid","score"]])

    if not all_scores:
        raise RerankError("No valid batches returned")

    # Combine and global sort
    scores = pd.concat(all_scores, ignore_index=True)
    merged = (
        candidates.assign(pid=candidates["pid"].astype(str))
        .merge(scores, on="pid", how="inner")
        .sort_values("score", ascending=False)
        .head(k)
    )
    return merged

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python llama_rerank.py paragraph.txt candidates.tsv")
        sys.exit(1)

    paragraph = Path(sys.argv[1]).read_text(encoding="utf-8")
    cand_df   = pd.read_csv(sys.argv[2], sep="\t", names=["pid","title","abstract"])

    try:
        top = rerank_batch(paragraph, cand_df, k=10)
        print(top[["pid","score"]])
    except RerankError as err:
        logging.error("Rerank failed: %s", err)