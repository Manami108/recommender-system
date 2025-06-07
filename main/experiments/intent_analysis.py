# --------------------- intent_analysis.py -------------------
"""Multi‑level intent extraction via CoT (uses sliding chunks)."""
import json, re, textwrap
from pathlib import Path
from typing import Dict, Any
from .config   import llama_gen, PROMPT_DIR
from .chunking import sliding_chunks

_REQUIRED = {"local_intent","paragraph_intent","global_intent",
             "keywords","research_domain","technologies"}
_PROMPT_TPL = (PROMPT_DIR/"intent_cot.prompt").read_text()
_JSON_RE = re.compile(r"\{[\s\S]*?\}\s*$", re.M)

def analyse(paragraph: str) -> Dict[str,Any]:
    joined = "\n-----\n".join(sliding_chunks(paragraph))
    prompt = _PROMPT_TPL.replace("{chunks}", joined)
    raw    = llama_gen()(prompt, max_new_tokens=700,
                         temperature=0.0, do_sample=False)[0]["generated_text"]
    match  = _JSON_RE.search(raw)
    data   = json.loads(match.group(0)) if match else {}
    missing = _REQUIRED - set(data)
    if missing:
        raise ValueError(f"Intent JSON missing: {missing}")
    return data
