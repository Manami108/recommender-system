import json
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ─────────────────────────────────────────────────────────────────────────────
# Load the few-shot CoT coherence/rerank prompt
# (make sure your prompt ends with the <<<PAR>>> marker and an <END> tag)
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_PATH     = Path(__file__).parent / "prompts" / "coherence_rerank.prompt"
PROMPT_TEMPLATE = PROMPT_PATH.read_text()

# ─────────────────────────────────────────────────────────────────────────────
# Initialize LLaMA-3 8B
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID  = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto"
)
gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1500,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    return_full_text=False    
)

# ─────────────────────────────────────────────────────────────────────────────
def extract_json_block(text: str) -> str:
    """
    Scan from the first '{' to the matching '}', returning that substring.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM output.")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    raise ValueError("Unbalanced JSON braces in LLM output.")
# ─────────────────────────────────────────────────────────────────────────────


def llm_rerank(paragraph: str, candidates: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    1) Inject the paragraph into your prompt at <<<PAR>>>.
    2) Generate CoT and extract the first JSON block.
    3) Score & rerank the candidate abstracts by topic overlap.
    """
    # 1) Build prompt
    # 1) Wrap your paragraph exactly as your prompt expects:
    wrapped = f"<TEXT>\n{paragraph.strip()}\n</TEXT>"

    # 2) Replace the one unique token—leave all {…} in the template untouched
    prompt = PROMPT_TEMPLATE.replace("<<<PAR>>>", paragraph.strip())

    # DEBUG: inspect what we actually send
    print("— DEBUG PROMPT START —")
    print(prompt)
    print("— DEBUG PROMPT END —")

    # 2) Generate + truncate at <END>
    raw = gen(prompt)[0]["generated_text"]
    raw = raw.split("<END>")[0]

    # 3) Extract JSON
    try:
        json_str = extract_json_block(raw)
        coh      = json.loads(json_str)
    except Exception as e:
        # Dump for debugging
        print("=== RAW OUTPUT ===")
        print(raw)
        print("=== END RAW OUTPUT ===")
        raise RuntimeError(f"JSON parse error: {e}")

    # 4) Gather window topics
    topics = {t.lower() for win in coh.get("windows", []) for t in win.get("topics", [])}

    # 5) Score candidates by overlap
    scores = []
    for _, row in candidates.iterrows():
        abst   = row.get("abstract", "") or ""
        tokens = {w.lower().strip(".,") for w in abst.split()}
        scores.append(len(topics & tokens))

    # 6) Return top-k
    df = candidates.copy()
    df["llm_score"] = scores
    return df.sort_values("llm_score", ascending=False).head(k)
