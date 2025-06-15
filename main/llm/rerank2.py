from __future__ import annotations
import json, logging, os, re, sys
from pathlib import Path
from typing import List

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ──────────────────────────── CONFIG ──────────────────────────── #
_MODEL_ID  = os.getenv("LLAMA_MODEL",  "meta-llama/Meta-Llama-3-8B-Instruct")
_DEVICE    = os.getenv("LLAMA_DEVICE", "auto")
MAX_GEN    = 8192
MAX_ABS_CH = 750
BATCH_SIZE = 3
MAX_POOL   = 40
TOK_HEAD   = 6144

# path to your prompt template
_PROMPT_PATH = Path(__file__).parent / "prompts" / "cars2.prompt"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ───────────────────────── LOAD MODEL ────────────────────────── #
_tok = AutoTokenizer.from_pretrained(_MODEL_ID, use_default_system_prompt=False)
_mdl = AutoModelForCausalLM.from_pretrained(_MODEL_ID, device_map=_DEVICE, torch_dtype="auto")
_gen = pipeline(
    "text-generation",
    model=_mdl,
    tokenizer=_tok,
    max_new_tokens=MAX_GEN,
    do_sample=True,
    temperature=0.3,
    top_p=0.9,
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

class RerankError(RuntimeError):
    pass

# ────────────────────────── CORE ────────────────────────────── #
def rerank_batch(
    paragraph: str,
    candidates: pd.DataFrame,
    *,
    k: int = 20,
    max_candidates: int = MAX_POOL,
    batch_size: int = BATCH_SIZE,
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

    batches: List[pd.DataFrame] = []
    for start in range(0, len(candidates), batch_size):
        part = candidates.iloc[start : start + batch_size]

        # build candidate block
        def trunc(t: str) -> str:
            return t if len(t) <= MAX_ABS_CH else t[:MAX_ABS_CH] + " …"

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

        # check length
        if len(_tok(prompt).input_ids) > TOK_HEAD:
            raise RerankError("Prompt too long; reduce batch size or truncate abstracts")

        raw_out = _gen(prompt)[0]["generated_text"]

        # strip any Llama special tokens
        raw = re.sub(r"<\|eot_id\|>.*$", "", raw_out, flags=re.DOTALL).strip()

        # ── robust extraction ─────────────────────────────────────────
        m = _JSON_RE.search(raw)
        if m:
            json_text = m.group(1)
        else:
            # fallback: first JSON array in the output
            arr_start = raw.find("[")
            arr_end   = raw.find("]", arr_start)
            if arr_start == -1 or arr_end == -1:
                raise RerankError(f"No JSON array found in LLM output:\n{raw[:300]}")
            json_text = raw[arr_start : arr_end + 1]

        # fix common pid-number bug → "pid": "123"
        json_text = re.sub(r'"pid"\s*:\s*([0-9]+)', r'"pid":"\1"', json_text)

        try:
            df = pd.DataFrame(json.loads(json_text))
        except Exception as e:
            raise RerankError(f"JSON parse failed: {e}\nSnippet:\n{json_text[:200]}")

        df.columns = [c.lower() for c in df.columns]
        if "pid" not in df:
            raise RerankError(f"No 'pid' in columns: {df.columns.tolist()}")
        if "rank" not in df:
            df["rank"] = range(1, len(df) + 1)
        else:
            df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype(int)

        batches.append(df[["pid", "rank"]])

    if not batches:
        raise RerankError("No valid batches returned")

    scores = pd.concat(batches, ignore_index=True)
    scores["pid"] = scores["pid"].astype(str)
    scores["rank"] = pd.to_numeric(scores["rank"], errors="coerce")

    candidates["pid"] = candidates["pid"].astype(str)
    merged = candidates.merge(scores, on="pid", how="inner")
    return merged.sort_values("rank").head(k)

# ────────────────────────── CLI ─────────────────────────────── #
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python llama_rerank.py paragraph.txt candidates.tsv")
        sys.exit(1)

    paragraph  = Path(sys.argv[1]).read_text(encoding="utf-8")
    cand_df    = pd.read_csv(sys.argv[2], sep="\t", names=["pid","title","abstract"])

    try:
        top = rerank_batch(paragraph, cand_df, k=10)
        print(top[["pid","rank"]])
    except RerankError as err:
        logging.error("Rerank failed: %s", err)
