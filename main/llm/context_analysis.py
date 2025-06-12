from pathlib import Path
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load your “context analysis” prompt
CTX_PROMPT = Path("prompts/context_analysis.prompt").read_text()

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
mdl = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
gen = pipeline("text-generation", model=mdl, tokenizer=tok,
               max_new_tokens=2000, do_sample=False,
               pad_token_id=tok.eos_token_id, return_full_text=False)

def analyze_paragraph_context(paragraph: str) -> dict:
    prompt = CTX_PROMPT.replace("<<<PAR>>>", paragraph.strip())
    raw = gen(prompt)[0]["generated_text"].split("<END>")[0]
    # extract the entire JSON object
    start, end = raw.find("{"), raw.rfind("}")+1
    ctx = json.loads(raw[start:end])
    return ctx
