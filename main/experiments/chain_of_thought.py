
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch, re, json

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
print("▶ Loading Llama-3 8B…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,  # I think this is memmory efficient, less VRAM usage!!
                device_map="auto"  #This line is to make sure model is loaded on GPU (however, in Manami's computer, it becomes CPU cuz simply, the my gpu cannot handle it. )
            )

# This is the generation pipeline 
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Extract concepts with chain of thought few-shot prompting. 
# https://www.promptingguide.ai/techniques/cot
# https://www.mercity.ai/blog-post/guide-to-chain-of-thought-prompting#what-is-chain-of-thought-prompting
def extract_context(paragraph: str) -> dict:
    prompt = f"""
You are an academic assistant an elite writing assistant for computer-science research.


TASK
Step 1 – THINK
  • Read the paragraph in <TEXT>.
  • Deliberate step-by-step inside the tags <COT> … </COT>.

Step 2 – EXTRACT six fields
  1) main_topic        – one short phrase
  2) subtopics         – up to 3 phrases (list)
  3) problem_statement – ONE sentence (≤ 30 tokens)
  4) technologies      – list concrete models / algorithms / datasets
  5) research_domain   – broad area (e.g. “machine learning”)
  6) user_intent       – 5-15 tokens (“survey X”, “find gaps in Y”, …)

Step 3 – OUTPUT
  • Write valid compact JSON only.
  • Keys must appear in the order shown above.

RULES
  • Do NOT invent papers, references, or details not implied by the text.
  • Outside <COT> and the final JSON, write nothing else.

EXAMPLES

Example 1
<COT>
• Identify domain terms: “BERT”, “GPT”, “bidirectional attention”.
• Core area ≈ “Transformer-based NLP” → main_topic.
• Sub-areas: pretraining, question answering, machine translation.
• Problem: traditional seq-models lack deep context.
• Tech list = {{BERT, GPT}}.
• Domain = NLP.
• Intent ≈ “survey modern transformer NLP work”.
</COT>
{{"main_topic":"transformer-based NLP",
  "subtopics":["pretraining","question answering","machine translation"],
  "problem_statement":"Sequence models before transformers struggled to capture long-range context in language tasks.",
  "technologies":["BERT","GPT"],
  "research_domain":"natural language processing",
  "user_intent":"survey modern transformer literature"}}

Example 2
<COT>
• Key concepts: “knowledge graph”, “entity linking”, “question answering”.
• main_topic = knowledge graphs.
• Subtopics: entity linking, QA, recommendation.
• Tech list none explicit → empty list.
• Problem stmt: structuring entities for downstream tasks.
• Domain: AI / NLP.
• Intent: discover KG applications.
</COT>
{{"main_topic":"knowledge graphs",
  "subtopics":["entity linking","question answering","recommendation systems"],
  "problem_statement":"Researchers need structured representations of entities and relations to enhance downstream tasks.",
  "technologies":[],
  "research_domain":"AI / information extraction",
  "user_intent":"discover KG application papers"}}

Example 3
<COT>
• Mentions: “graph neural networks”, “GCN”, “GAT”, “MPNN”.
• main_topic = graph neural networks.
• Subtopics: node classification, link prediction, molecular property.
• Tech list = {{GCN, GAT, MPNN}}.
• Problem stmt derived.
• Domain: machine learning.
• Intent: compare GNN variants.
</COT>
{{"main_topic":"graph neural networks",
  "subtopics":["node classification","link prediction","molecular property prediction"],
  "problem_statement":"Existing deep-learning methods need adaptation to non-Euclidean graph data structures.",
  "technologies":["Graph Convolutional Network","Graph Attention Network","Message Passing Neural Network"],
  "research_domain":"machine learning",
  "user_intent":"compare GNN variants"}}

PARAGRAPH
<TEXT>
{paragraph}
</TEXT>

# After thinking in <COT>, output your JSON on a new line.
"""

# Max new tokens are set to 600 to limit the LLM's answer (but the token is in addition to the input)
# temperature tells what kind of output. For example, 0.0 tells always same output for the same input but 1.0 has balanced randomness. so 0.0 tells conssistent and accurate answers
# Sample also tells the randomness. so if i set sample = True, then temerature tells how much randomness i want.
    raw = generator(
        prompt,
        max_new_tokens=1000,
        temperature=0.0,
        do_sample=False
    )[0]["generated_text"]

    # 2) Locate the *last* `{` and the *last* `}` in raw
    last_open = raw.rfind("{")
    last_close = raw.rfind("}")
    if last_open == -1 or last_close == -1 or last_close < last_open:
        raise ValueError("Extractor failed – could not find a well-formed JSON block.\n\nRaw output:\n" + raw)

    last_json_str = raw[last_open : last_close + 1]

    # 3) Parse it
    try:
        data = json.loads(last_json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON returned:\n{last_json_str}\n\nRaw output:\n{raw}") from e

    # 4) Sanity check required keys
    required = {
        "main_topic",
        "subtopics",
        "problem_statement",
        "technologies",
        "research_domain",
        "user_intent"
    }
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Missing keys in returned JSON: {missing}\n\nReturned JSON:\n{last_json_str}")

    return data


# Demo
if __name__ == "__main__":
    paragraph = (
        "Recently, as advanced natural language processing techniques, "
        "Large Language Models (LLMs) with billion parameters have generated "
        "large impacts on various research fields such as NLP, Computer Vision, "
        "and Molecule Discovery. Technically most existing LLMs are transformer-"
        "based models pre-trained on a vast amount of textual data from diverse sources. "
        "As the parameter size of LLMs continues to scale, recent studies indicated that LLMs "
        "can lead to the emergence of remarkable capabilities. More specifically, LLMs demonstrate "
        "powerful abilities in language understanding and generation, enabling them to better "
        "comprehend human intentions. Moreover, LLMs exhibit impressive generalization and "
        "reasoning, often applying learned knowledge to new tasks with few-shot demonstrations."
    )

    ctx = extract_context(paragraph)
    print("Extracted context:")
    for k, v in ctx.items():
        print(f"{k:18}: {v}")

