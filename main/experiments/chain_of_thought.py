# chain_of_thought.py

from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
_tokenizer = AutoTokenizer.from_pretrained(MODEL)
_model     = AutoModelForCausalLM.from_pretrained(
                MODEL,
                torch_dtype="auto",
                device_map="auto"
            )
_gen = pipeline("text-generation", model=_model, tokenizer=_tokenizer)

# (Make sure paragraph_context.prompt only contains your examples up to the JSON,
#  and does NOT include extra “# Your output here…” comment lines after that.)
PROMPT = (Path(__file__).parent/"prompts"/"paragraph_context.prompt").read_text()

_REQUIRED_KEYS = {
    "main_topic",
    "subtopics",
    "problem_statement",
    "technologies",
    "research_domain",
    "user_intent"
}

def _last_json_block(text: str) -> str:
    """
    Grab only the substring from the final '{' up to the final '}' in the LLM output.
    """
    lo = text.rfind("{")
    hi = text.rfind("}")
    if lo == -1 or hi == -1 or hi < lo:
        raise ValueError("No JSON block found in model output.")
    return text[lo : hi + 1]

def extract_context(paragraph: str) -> dict:
    """
    1) Send the few-shot CoT prompt to LLaMA-3.
    2) Extract only the trailing {...} block (using rfind),
       ignoring any extra '#' comments or repeated examples.
    3) Parse it into JSON, verify required keys.
    """
    prompt = PROMPT.replace("{paragraph}", paragraph)

    raw = _gen(
        prompt,
        max_new_tokens=1000,
        temperature=0.0,
        do_sample=False
    )[0]["generated_text"]

    try:
        json_str = _last_json_block(raw)
    except ValueError:
        # If we can’t find a clean JSON block, print the raw output for debugging
        print("\n[chain_of_thought] ERROR: No JSON found. Raw output:")
        print("─" * 60)
        print(raw)
        print("─" * 60)
        raise

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # If it looks like JSON but fails to parse, show that block so we can inspect it
        print("\n[chain_of_thought] ERROR: Invalid JSON block:")
        print("─" * 60)
        print(json_str)
        print("─" * 60)
        raise ValueError("Invalid JSON returned by model")

    missing = _REQUIRED_KEYS - set(data.keys())
    if missing:
        print("\n[chain_of_thought] ERROR: JSON missing keys:", missing)
        print("Returned JSON:", data)
        raise ValueError(f"Missing keys in returned JSON: {missing}")

    return data
