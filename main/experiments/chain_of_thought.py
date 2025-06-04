from pathlib import Path
import json, torch, re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---------- 1.  load model once -------------------------------------------
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
print("▶ loading Llama-3 8B …")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
_model     = AutoModelForCausalLM.from_pretrained(
                 MODEL_ID,
                 torch_dtype=torch.float16,
                 device_map="auto"
             )
_generator  = pipeline("text-generation", model=_model, tokenizer=_tokenizer)

# ---------- 2.  load prompt template (disk -> str) ------------------------
PROMPT_FILE = Path(__file__).parent / "prompts" / "paragraph_context.prompt"
PROMPT_TEMPLATE = PROMPT_FILE.read_text(encoding="utf-8")

# ---------- 3.  helper ----------------------------------------------------
_REQUIRED_KEYS = {
    "main_topic", "subtopics", "problem_statement",
    "technologies", "research_domain", "user_intent"
}

def _last_json_block(text: str) -> str:
    """Return substring from the last '{' to the matching final '}'."""
    lo, hi = text.rfind("{"), text.rfind("}")
    if lo == -1 or hi == -1 or hi < lo:
        raise ValueError("No JSON block at end of model output.")
    return text[lo:hi + 1]

# ---------- 4.  public API -----------------------------------------------
def extract_context(paragraph: str, max_new_tokens: int = 600) -> dict:
    """Return context dict for a paragraph using the shared Llama generator."""
    prompt = PROMPT_TEMPLATE.format(paragraph=paragraph)

    raw = _generator(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False
    )[0]["generated_text"]

    json_str = _last_json_block(raw)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM produced invalid JSON:\n{json_str}") from e

    missing = _REQUIRED_KEYS - data.keys()
    if missing:
        raise ValueError(f"Missing keys {missing} in JSON:\n{data}")

    return data
