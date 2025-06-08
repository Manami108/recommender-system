import json
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ─────────────────────────────────────────────────────────────────────────────
# Load the few-shot CoT coherence/rerank prompt
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_PATH = Path(__file__).parent / "prompts" / "coherence_rerank.prompt"
PROMPT_TEMPLATE = PROMPT_PATH.read_text()

# ─────────────────────────────────────────────────────────────────────────────
# Initialize LLaMA-3 8B (meta-llama/Meta-Llama-3-8B-Instruct)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto"
)
gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1500,
    temperature=0.0,
    do_sample=False
)

# ─────────────────────────────────────────────────────────────────────────────
# Reranker: apply CoT analysis and produce an llm_score
# ─────────────────────────────────────────────────────────────────────────────
def llm_rerank(paragraph: str, candidates: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    Use chain-of-thought coherence analysis on the paragraph,
    then score & rerank papers based on topic overlap.

    Args:
        paragraph: the text to analyze
        candidates: DataFrame with ['pid','title','abstract']
        k: number of top candidates to return

    Returns:
        DataFrame with added 'llm_score', sorted desc, top-k rows
    """
    # 1) Build prompt
    prompt = PROMPT_TEMPLATE.replace("<<<PARAGRAPH>>>", paragraph)

    # 2) Generate CoT JSON
    raw = gen(prompt)[0]["generated_text"]
    lo, hi = raw.rfind("{"), raw.rfind("}")
    try:
        coh = json.loads(raw[lo:hi+1])
    except Exception as e:
        raise RuntimeError(f"Parsing JSON failed: {e}\nRaw output:\n{raw}")

    # 3) Extract window topics
    topics = {t.lower() for win in coh.get('windows', []) for t in win.get('topics', [])}

    # 4) Score each candidate by overlap
    scores = []
    for _, row in candidates.iterrows():
        abst = row.get('abstract', '') or ''
        tokens = {w.lower().strip('.,') for w in abst.split()}
        scores.append(len(topics & tokens))

    # 5) Attach scores & return top-k
    df = candidates.copy()
    df['llm_score'] = scores
    return df.sort_values('llm_score', ascending=False).head(k)
