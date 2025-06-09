# rerank_llm.py
from __future__ import annotations
import json, re
from pathlib import Path
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# ── 1. model & generator ───────────────────────────────────────────
_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
_tok  = AutoTokenizer.from_pretrained(_MODEL_ID)
_mdl  = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID,
            device_map="auto",
            torch_dtype="auto")
_gen  = pipeline("text-generation",
                 model=_mdl, tokenizer=_tok,
                 max_new_tokens=2000,
                 do_sample=False,
                 pad_token_id=_tok.eos_token_id,
                 return_full_text=False)

# ── 2. load prompt template once ──────────────────────────────────
_SCORE_TMPL = Path("prompts/coherence_score.prompt").read_text()

_JSON_RE = re.compile(r"<RESULT>[\\s\\S]*?\\[.*?\\][\\s\\S]*?</RESULT>")

def llm_contextual_rerank(
        paragraph: str,
        candidates: pd.DataFrame,
        k: int = 10
) -> pd.DataFrame:
    """Return top-k candidates with LLM-derived coherence score."""

    # Build candidate block (same order as DataFrame)
    cand_lines = [
        f"{row.pid}\nTitle: {row.title}\nAbstract: {row.abstract.strip()}"
        for _, row in candidates.iterrows()
    ]
    cand_block = "\n\n".join(cand_lines)

    # Fill template
    prompt = (_SCORE_TMPL
              .replace("<<<PARAGRAPH>>>", paragraph.strip())
              .replace("<<<CANDIDATES>>>", cand_block))

    raw = _gen(prompt)[0]["generated_text"]
    m   = _JSON_RE.search(raw)
    if not m:
        raise ValueError("LLM did not return a JSON array.")

    score_df = pd.DataFrame(json.loads(m.group(0)))
    ranked   = candidates.merge(score_df, on="pid", how="inner")
    ranked   = ranked.sort_values("final_score", ascending=False)

    return ranked.head(k)
