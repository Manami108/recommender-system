from __future__ import annotations
import json, re, logging
from pathlib import Path

import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)

# ── 1. model & generator ───────────────────────────────────────────
_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
_tok  = AutoTokenizer.from_pretrained(_MODEL_ID)
_mdl  = AutoModelForCausalLM.from_pretrained(
    _MODEL_ID,
    device_map="auto",
    torch_dtype="auto"
)
_gen = pipeline(
  "text-generation",
  model=_mdl,
  tokenizer=_tok,
  max_new_tokens=1000,
  do_sample=False,
  temperature=0,
  top_p=1,
  pad_token_id=_tok.eos_token_id,
  return_full_text=False,
  verbosity="error",
)

# ── 2. load prompt template once ──────────────────────────────────
_SCORE_TMPL = Path("prompts/cars3.prompt").read_text()

_JSON_RE     = re.compile(r"<RESULT>\s*(\[[\s\S]*?\])\s*</RESULT>", re.MULTILINE)
_FENCED_JSON = re.compile(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", re.MULTILINE)

def batch_df(df: pd.DataFrame, batch_size: int):
    """Yield successive DataFrame chunks of size batch_size."""
    for i in range(0, len(df), batch_size):
        yield df.iloc[i : i + batch_size]

def llm_contextual_rerank(
    paragraph: str,
    candidates: pd.DataFrame,
    k: int = 10,
    max_candidates: int = 40,
    batch_size: int | None = None,
) -> pd.DataFrame:
    """
    Return top-k candidates with LLM-derived coherence scores.
    - max_candidates: cap # of candidates before rerank
    - batch_size: if None or >= num candidates → run one pass
    """
    # 0) limit candidates for memory efficiency
    if len(candidates) > max_candidates:
        candidates = candidates.iloc[:max_candidates].copy()

    # 1) determine batch size
    if batch_size is None or batch_size >= len(candidates):
        batch_size = len(candidates)

    all_scores: list[pd.DataFrame] = []

    for batch in batch_df(candidates, batch_size):
        # Build the candidate block
        cand_lines = []
        for _, row in batch.iterrows():
            abs_txt = (row.abstract or "").strip()
            cand_lines.append(
                f"{row.pid}\nTitle: {row.title}\nAbstract: {abs_txt}"
            )
        cand_block = "\n\n".join(cand_lines)

        # Fill prompt
        prompt = (
            _SCORE_TMPL
            .replace("<<<PARAGRAPH>>>", paragraph.strip())
            .replace("<<<CANDIDATES>>>", cand_block)
        )

        # Single generation call per batch
        raw = _gen(prompt)[0]["generated_text"]

        # Try to extract JSON
        m = _JSON_RE.search(raw) or _FENCED_JSON.search(raw)
        if m:
            json_text = m.group(1)
        else:
            # fallback
            start = raw.find('[')
            end   = raw.rfind(']') + 1
            if start == -1 or end == 0:
                logging.warning("No JSON array found in LLM output. Raw:\n%s", raw)
                continue
            json_text = raw[start:end]

        try:
            batch_scores = pd.DataFrame(json.loads(json_text))
        except Exception as e:
            logging.warning("JSON parsing failed: %s\nJSON slice:\n%s", e, json_text)
            continue

        all_scores.append(batch_scores)

    if not all_scores:
        raise ValueError("No valid scores returned from any batch.")

    # Merge all batches and pick top-k by rank
    scores_df = pd.concat(all_scores, ignore_index=True)
    scores_df["rank"] = pd.to_numeric(scores_df["rank"], errors="coerce")
    ranked = candidates.merge(scores_df, on="pid", how="inner")
    ranked = ranked.sort_values("rank", ascending=True)

    return ranked.head(k)
