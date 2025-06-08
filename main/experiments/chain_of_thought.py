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
_gen = pipeline(
    "text-generation",
    model=_model,
    tokenizer=_tokenizer,
    return_full_text=False,
)

# ─────────────────────────────────────────────
# 1. Prompt loading helpers
# ─────────────────────────────────────────────
PROMPTS_DIR = Path(__file__).parent / "prompts"
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

def _load_prompt(name: str, fallback: str) -> str:
    path = PROMPTS_DIR / name
    try:
        return path.read_text()
    except FileNotFoundError:
        return fallback

LOCAL_PROMPT = _load_prompt(
    "local_intent.prompt",
    "INPUT:\n{chunk}\nOUTPUT: {\"local_intent\":null, \"local_coherence\":null}"
)
PARAGRAPH_PROMPT = _load_prompt(
    "paragraph_intent.prompt",
    "INPUT:\n{paragraph}\nOUTPUT: {\"paragraph_intent\":null, \"paragraph_cohesion\":null}"
)
GLOBAL_PROMPT = _load_prompt(
    "global_coherence.prompt",
    "INPUT:\n{payload}\nOUTPUT: {\"global_overview\":null, \"global_coherence\":null}"
)

KEYS_LOCAL = {"local_intent", "local_coherence"}
KEYS_PARAGRAPH = {"paragraph_intent", "paragraph_cohesion"}
KEYS_GLOBAL = {"global_overview", "global_coherence"}

# ─────────────────────────────────────────────
# 2. Utility – extract JSON from LLM output robustly
# ─────────────────────────────────────────────

def _extract_json(raw: str) -> Dict:
    # find the first '{' in the generated text
    start = raw.find("{")
    if start == -1:
        print("=== LLM output (no JSON found) ===\n" + raw + "\n=== end ===")
        raise ValueError("No JSON block found in model output.")

    # walk through to find matching closing brace
    level = 0
    end = None
    for i, ch in enumerate(raw[start:], start):
        if ch == '{':
            level += 1
        elif ch == '}':
            level -= 1
            if level == 0:
                end = i
                break

    if end is None:
        print("=== LLM output (unbalanced braces) ===\n" + raw + "\n=== end ===")
        raise ValueError("Unbalanced braces in model output.")

    json_str = raw[start : end + 1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print("=== Failed to parse JSON: ===\n" + json_str + "\nError: " + str(e))
        raise

# ─────────────────────────────────────────────
# 3. Analysis functions
# ─────────────────────────────────────────────

def analyze_local(chunks: List[str]) -> List[Dict]:
    signals: List[Dict] = []
    for idx, ch in enumerate(chunks):
        prompt = LOCAL_PROMPT.replace("{chunk}", ch)
        result = _gen(prompt, max_new_tokens=512)
        out = result[0]
        raw = out.get("generated_text", "")
        data = _extract_json(raw)
        if not KEYS_LOCAL.issubset(data.keys()):
            raise ValueError(f"Local JSON missing {KEYS_LOCAL - data.keys()} in chunk {idx}")
        signals.append(data)
    return signals

def analyze_paragraph(paragraph: str) -> Dict:
    prompt = PARAGRAPH_PROMPT.replace("{paragraph}", paragraph)
    result = _gen(prompt, max_new_tokens=512)
    out = result[0]
    raw = out.get("generated_text", "")
    data = _extract_json(raw)
    if not KEYS_PARAGRAPH.issubset(data.keys()):
        raise ValueError(f"Paragraph JSON missing {KEYS_PARAGRAPH - data.keys()}")
    return data

def analyze_global(local_signals: List[Dict], paragraph_signal: Dict) -> Dict:
    payload = json.dumps({"local": local_signals, "paragraph": paragraph_signal}, ensure_ascii=False)
    prompt = GLOBAL_PROMPT.replace("{payload}", payload)
    result = _gen(prompt, max_new_tokens=512)
    out = result[0]
    raw = out.get("generated_text", "")
    data = _extract_json(raw)
    if not KEYS_GLOBAL.issubset(data.keys()):
        raise ValueError(f"Global JSON missing {KEYS_GLOBAL - data.keys()}")
    return data

def run_cot_analysis(chunks: List[str], paragraph: str) -> Dict:
    local_sig = analyze_local(chunks)
    para_sig = analyze_paragraph(paragraph)
    global_sig = analyze_global(local_sig, para_sig)
    return {
        "local_signals": local_sig,
        "paragraph_signal": para_sig,
        "global_signal": global_sig,
    }

# ─────────────────────────────────────────────
# 4. Demo
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from chunking import clean_text, chunk_tokens

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

    cleaned = clean_text(demo_para)
    chunks = chunk_tokens(cleaned, _tokenizer, win=128, stride=64)

    print(f"Found {len(chunks)} token chunks → running CoT analysis …", file=sys.stderr)
    signals = run_cot_analysis(chunks, cleaned)
    print(json.dumps(signals, indent=2, ensure_ascii=False))