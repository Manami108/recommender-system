from pathlib import Path
import json, torch, re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
print("▶ loading Llama-3 8B …")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
_model     = AutoModelForCausalLM.from_pretrained(
                 MODEL_ID,
                 torch_dtype=torch.float16,  # I think this is memmory efficient, less VRAM usage!!
                 device_map="auto" #This line is to make sure model is loaded on GPU (however, in Manami's computer, it becomes CPU cuz simply, the my gpu cannot handle it. )
             )


# This is the generation pipeline 
_generator  = pipeline("text-generation", model=_model, tokenizer=_tokenizer)

# Now, prompt file is in different folder. so I need to call it first. 
# https://www.promptingguide.ai/techniques/cot
# https://www.mercity.ai/blog-post/guide-to-chain-of-thought-prompting#what-is-chain-of-thought-prompting
PROMPT_FILE = Path(__file__).parent / "prompts" / "paragraph_context.prompt"
PROMPT_TEMPLATE = PROMPT_FILE.read_text(encoding="utf-8")


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


# Max new tokens are set to 600 to limit the LLM's answer (but the token is in addition to the input)
# temperature tells what kind of output. For example, 0.0 tells always same output for the same input but 1.0 has balanced randomness. so 0.0 tells conssistent and accurate answers
# Sample also tells the randomness. so if i set sample = True, then temerature tells how much randomness i want.
def extract_context(paragraph: str) -> dict:
    """Return context dict for a paragraph using the shared Llama generator."""
    prompt = PROMPT_TEMPLATE.format(paragraph=paragraph)

    raw = _generator(
        prompt,
        max_new_tokens=1000,
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
