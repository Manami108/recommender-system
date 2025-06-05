# chain_of_thought.py
from pathlib import Path
import json, re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
_tokenizer = AutoTokenizer.from_pretrained(MODEL)
_model     = AutoModelForCausalLM.from_pretrained(
                MODEL, torch_dtype="auto", device_map="auto")
_gen       = pipeline("text-generation", model=_model, tokenizer=_tokenizer)

PROMPT = (Path(__file__).parent/"prompts"/"paragraph_context.prompt").read_text()

_KEYS = {"main_topic","subtopics","problem_statement",
         "technologies","research_domain","user_intent"}

def _json_tail(txt:str) -> str:
    lo, hi = txt.rfind("{"), txt.rfind("}")
    if lo == -1 or hi < lo: raise ValueError("JSON not found")
    return txt[lo:hi+1]

def extract_context(paragraph:str)->dict:
    out = _gen(PROMPT.format(paragraph=paragraph),
               max_new_tokens=800, temperature=0.0,
               do_sample=False)[0]["generated_text"]
    data = json.loads(_json_tail(out))
    if not _KEYS.issubset(data): raise ValueError("Missing keys")
    return data
