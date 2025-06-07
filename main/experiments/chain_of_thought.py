"""
chain_of_thought_analysis.py
───────────────────────────
Stage‑1 processing for the new pipeline
  ▸ Local intent / coherence per sliding‑window chunk
  ▸ Paragraph‑level intent & cohesion
  ▸ Global document‑level overview + coherence

Everything is wrapped in three helper functions and a `run_cot_analysis` orchestrator.
A demo runnable with `python chain_of_thought_analysis.py` is included at the bottom.

Prompts live under ./prompts/ as plain‑text files:
  prompts/
    ├─ local_intent.prompt
    ├─ paragraph_intent.prompt
    └─ global_coherence.prompt

Each file contains your few‑shot examples followed by a single "INPUT:" token where
`{chunk}` or `{paragraph}` will be substituted, and ends with "OUTPUT:" so the model
responds only with a JSON object.  The loader below falls back to a minimal one‑line
prompt if the file is missing so the script remains importable.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from typing import List, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# ─────────────────────────────────────────────
# 0. Model / tokenizer / generator (loaded once)
# ─────────────────────────────────────────────
_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
_model     = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)
_gen = pipeline("text-generation", model=_model, tokenizer=_tokenizer)

# ─────────────────────────────────────────────
# 1. Prompt loading helpers
# ─────────────────────────────────────────────
PROMPTS_DIR = Path(__file__).parent / "prompts"
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)  # ensure directory exists

def _load_prompt(name: str, fallback: str) -> str:
    path = PROMPTS_DIR / name
    try:
        return path.read_text()
    except FileNotFoundError:
        return fallback

# Minimal fall‑back prompts (overwrite with real few‑shot examples!)
LOCAL_PROMPT      = _load_prompt(
    "local_intent.prompt",
    "INPUT:\n{chunk}\nOUTPUT: {\"local_intent\":null, \"local_coherence\":null}"
)
PARAGRAPH_PROMPT  = _load_prompt(
    "paragraph_intent.prompt",
    "INPUT:\n{paragraph}\nOUTPUT: {\"paragraph_intent\":null, \"paragraph_cohesion\":null}"
)
GLOBAL_PROMPT     = _load_prompt(
    "global_coherence.prompt",
    "INPUT:\n{payload}\nOUTPUT: {\"global_overview\":null, \"global_coherence\":null}"
)

# Expected keys in the JSON blocks
KEYS_LOCAL      = {"local_intent", "local_coherence"}
KEYS_PARAGRAPH  = {"paragraph_intent", "paragraph_cohesion"}
KEYS_GLOBAL     = {"global_overview", "global_coherence"}

# ─────────────────────────────────────────────
# 2. Utility – extract trailing JSON from LLM output
# ─────────────────────────────────────────────

def _extract_json(raw: str) -> Dict:
    lo = raw.rfind("{")
    hi = raw.rfind("}")
    if lo == -1 or hi == -1 or hi < lo:
        raise ValueError("No JSON block found in model output.\nLLM said:\n" + raw)
    return json.loads(raw[lo : hi + 1])

# ─────────────────────────────────────────────
# 3. Analysis functions
# ─────────────────────────────────────────────

def analyze_local(chunks: List[str]) -> List[Dict]:
    """Run the local‑intent prompt on each chunk."""
    signals = []
    for idx, ch in enumerate(chunks):
        prompt = LOCAL_PROMPT.replace("{chunk}", ch)
        raw = _gen(prompt, max_new_tokens=512, temperature=0.0, do_sample=False)[0]["generated_text"]
        data = _extract_json(raw)
        missing = KEYS_LOCAL - data.keys()
        if missing:
            raise ValueError(f"Local JSON missing {missing} in chunk {idx}")
        signals.append(data)
    return signals

def analyze_paragraph(paragraph: str) -> Dict:
    prompt = PARAGRAPH_PROMPT.replace("{paragraph}", paragraph)
    raw = _gen(prompt, max_new_tokens=512, temperature=0.0, do_sample=False)[0]["generated_text"]
    data = _extract_json(raw)
    missing = KEYS_PARAGRAPH - data.keys()
    if missing:
        raise ValueError(f"Paragraph JSON missing {missing}")
    return data

def analyze_global(local_signals: List[Dict], paragraph_signal: Dict) -> Dict:
    payload_json = json.dumps({"local": local_signals, "paragraph": paragraph_signal}, ensure_ascii=False)
    prompt = GLOBAL_PROMPT.replace("{payload}", payload_json)
    raw = _gen(prompt, max_new_tokens=512, temperature=0.0, do_sample=False)[0]["generated_text"]
    data = _extract_json(raw)
    missing = KEYS_GLOBAL - data.keys()
    if missing:
        raise ValueError(f"Global JSON missing {missing}")
    return data

def run_cot_analysis(chunks: List[str], paragraph: str) -> Dict:
    local  = analyze_local(chunks)
    para   = analyze_paragraph(paragraph)
    global_sig = analyze_global(local, para)
    return {"local_signals": local, "paragraph_signal": para, "global_signal": global_sig}

# ─────────────────────────────────────────────
# 4. Demo  (python chain_of_thought_analysis.py)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Inline demo paragraph
    demo_para = """
This paper accordingly proposes a novel Context-guided Triple Matching (CTM),
while the third component missing from the pairwise matching is adopted as a prior context.
The proposed triple matching is present as a hierarchical attention flow to adequately capture
the semantic relationship. Specifically, given a candidate triple, we first employ (any) one
component from the triple as the prior context. Then we apply the bidirectional attention to
calculate the correlation between context and the other two components separately. Afterwards,
another attention layer is utilized to leverage two above correlations to form an aggregated
context-aware representation. In this way, the model is able to gather more comprehensive
semantic relationship for the triple, according to the selected context. Similarly, we enumerate
the other two components (from the triple) and cast as the prior context to repeat the same
attention flow. Finally, a fully-connected layer is employed for all formed context-aware
representations to estimate the matching score. In addition to the triple matching, we also
consider to adopt a contrastive regularization in capturing the subtle semantic differences among
answer candidates. The aim is to maximize the similarity of features from correct triple(s) while
pushing away that of distractive ones, that has been neglected by existing methods.
    """.strip()

    # Stage‑0  preprocess → sliding‑window token chunks
    from chunking import clean_text, chunk_tokens

    cleaned = clean_text(demo_para)
    chunks  = chunk_tokens(cleaned, _tokenizer, win=128, stride=64)

    print("Found", len(chunks), "token chunks → running CoT analysis …", file=sys.stderr)

    signals = run_cot_analysis(chunks, cleaned)
    print(json.dumps(signals, indent=2, ensure_ascii=False))
