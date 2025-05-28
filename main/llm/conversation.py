

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load model and tokenizer
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Interactive loop
print("TinyLlama Chat is ready! Type 'exit' to quit.\n")

history = ""

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Append user input
    history += f"<|user|>\n{user_input}\n<|assistant|>\n"

    # Generate a response
    output = pipe(history, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9)
    full_output = output[0]["generated_text"]

    # Extract only the assistant's response
    reply = full_output[len(history):].split("<|user|>")[0].strip()

    print(f"Assistant: {reply}\n")

    # Append assistant's reply to history
    history += f"{reply}\n"
