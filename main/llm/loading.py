
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
prompt = "What is the purpose of a scientific paper recommender system?"

response = pipe(prompt, max_new_tokens=100)
print(response[0]["generated_text"])
