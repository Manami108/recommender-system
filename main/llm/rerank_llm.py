# This code is to rerank those candidates using LLM based on how relevant each paper is to a given input paragraph. 

from __future__ import annotations
import json, logging, os, re, sys
from pathlib import Path
from typing import List
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig

# config
_MODEL_ID  = os.getenv("LLAMA_MODEL",  "meta-llama/Meta-Llama-3.1-8B-Instruct")

# path to prompt
# https://www.promptingguide.ai/jp/techniques/cot
# https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/
# https://medium.com/@tahirbalarabe2/prompt-engineering-with-llama-3-3-032daa5999f7
# https://www.kaggle.com/code/manojsrivatsav/prompt-engineering-with-llama-3-1-8b

_PROMPT_PATH = Path(__file__).parent / "prompts" / "working1.prompt"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# QLoRa
# https://reinforz.co.jp/bizmedia/13036/
# https://note.com/npaka/n/na506c63b8cc9
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,             
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
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
BATCH_SIZE = 3 # how many candidates per LLM call
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
def rerank_batch(
    paragraph: str,
    candidates: pd.DataFrame,
    *,
    k: int = 20, # This is how many final top papers to return 
    max_candidates: int = MAX_POOL, # drop any beyond this before batching.
    batch_size: int = BATCH_SIZE, # how many candidates per LLM API call.
) -> pd.DataFrame:
    # Return best k candidates
    if len(candidates) > max_candidates:
        candidates = candidates.iloc[:max_candidates].copy()

    def wrap_prompt(body: str) -> str:
        # wrap with begin_of_text and end_of_turn
        return "<|begin_of_text|>\n" + body.strip() + "\n<|eot_id|>"

    # Loops over the candidates in chunks of batch_size (e.g. 3 at a time).
    all_scores: List[pd.DataFrame] = []
    for start in range(0, len(candidates), batch_size):
        part = candidates.iloc[start : start + batch_size]

        # Concatenates each candidate’s ID, title, and a truncated abstract (so you don’t blow past the token limit).
        # Now, I am not truncating anymore
        cand_block = "\n\n".join(
            f"PID: {r.pid}\nTitle: {r.title}\nAbstract: {r.abstract}"
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
        
        # This is activated when you do normal zeroshot learning 
        # adjust token size 
        # tokens_per_item = 12          
        # max_gen_this_call = tokens_per_item * len(part) + 32
        # raw_out = _gen(
        #     prompt,
        #     max_new_tokens=max_gen_this_call,
        #     do_sample=False,
        #     pad_token_id=_tok.eos_token_id,
        #     return_full_text=False,
        # )[0]["generated_text"]
        # print("----- LLM RAW -----\n", raw_out[:400], "\n-------------------")

        # This is activated when you do chain of thought prompting 
        raw_out = _gen(prompt)[0]["generated_text"]   # uses global MAX_GEN
        # torch.cuda.empty_cache()

        
        raw = re.sub(r"<\|(?:eot_id|eom_id)\|>.*$", "", raw_out, flags=re.DOTALL).strip()
        # print("prompt tokens:", len(_tok(prompt).input_ids))
        # print("max_new_tokens:", max_gen_this_call)
        # print(raw_out)   # full, or at least first 800 chars
        # print("[checkpoint] after prompt")      # already prints prompt
        # torch.cuda.synchronize()
        # print("[checkpoint] before generate")
        # raw_out = _gen(prompt)[0]["generated_text"]
        # print("[checkpoint] after generate")    # you will never see this if crash is here
        # torch.cuda.synchronize()

        # Extract JSON object containing pid + score
        m = _JSON_RE.search(raw)
        if m:
            json_text = m.group(1)                   
        else:
            # fallback: collect every standalone {...} object
            objs = re.findall(r'\{[^{}]+\}', raw)
            if not objs:
                raise RerankError(
                    f"No JSON object found in LLM output (first 300 chars):\n{raw[:300]}")
            json_text = "[" + ",".join(objs) + "]"    # wrap → valid JSON array

        # Remove any stray trailing commas before ] or }
        tidy = re.sub(r",\s*(?=[\]\}])", "", json_text)            # remove trailing commas
        tidy = re.sub(r'"pid"\s*:\s*([0-9]+)', r'"pid":"\1"', tidy) # ensure pid quoted

        # Parse
        try:
            rec = json.loads(tidy)
        except Exception as e:
            raise RerankError(f"JSON parse failed: {e}\nSnippet:\n{tidy[:200]}")

        df = pd.DataFrame(rec)
        if {"pid", "score"} - set(df.columns):
            raise RerankError("Expected 'pid' and 'score' fields in JSON.")

        df["pid"] = df["pid"].astype(str)
        df["score"] = pd.to_numeric(df["score"], errors="coerce")

        all_scores.append(df[["pid","score"]])

    if not all_scores:
        raise RerankError("No valid batches returned")

    # Combine and global sort
    scores = pd.concat(all_scores, ignore_index=True)
    # print("\n[DEBUG] raw scores:\n", scores.to_string(index=False))
    result = (
        candidates.assign(pid=candidates["pid"].astype(str))
        .merge(scores, on="pid", how="inner")
        .sort_values("score", ascending=False)
        .head(k)
    )

    logging.debug("Top-k after rerank:\n%s", result[["pid", "score"]].to_string(index=False))
    return result

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