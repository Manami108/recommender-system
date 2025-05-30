# This tries weight scoring method but have the error so need to reformulated

import os, re, json, warnings, pandas as pd, torch
from neo4j import GraphDatabase
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import re, json
import torch
# This code does load model (huggingdace)

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
print("▶ Loading LLaMA 3-8B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16, # I think this is memmory efficient, less VRAM usage!!
    device_map="auto" #This line is to make sure model is loaded on GPU (however, in Manami's computer, it becomes CPU cuz simply, the my gpu cannot handle it. )
)

# This is few-shot prompt engineering. 
PROMPT_TEMPLATE = r"""
You are an academic assistant for computer-science papers.
You are an academic assistant of computer science field. Extract the most important research concepts (keywords of the from the paragraph wrapped in <TEXT> tags.
Extract some unique terms rather than common words in computer science paper paragraph. 
Add a `"weight"` (0.5–1.0) that reflects each concept’s importance.
Return **only** JSON of the form:
{{"concepts":[{{"term":"...", "weight":0.87}}, ...]}}

Example 1:
<TEXT>
Transformer-based architectures, like BERT and GPT, have revolutionized NLP by enabling bidirectional attention and large-scale pretraining.
These models achieve state-of-the-art results in tasks such as question answering, machine translation, and text summarization.
</TEXT>
Expected output:
{{"concepts":[
  {{"term":"transformer",                       "weight":0.95}},
  {{"term":"BERT",                              "weight":0.90}},
  {{"term":"GPT",                               "weight":0.90}},
  {{"term":"bidirectional attention",           "weight":0.80}},
  {{"term":"large-scale pretraining",           "weight":0.75}},
  {{"term":"question answering",                "weight":0.70}},
  {{"term":"machine translation",               "weight":0.70}},
  {{"term":"text summarization",                "weight":0.65}}
]}}

Example 2:
<TEXT>
Knowledge graphs represent entities and their relations as a structured graph. They are widely used in tasks like entity linking, question answering, and recommendation systems.
</TEXT>
Expected output:
{{"concepts":[
  {{"term":"knowledge graph",   "weight":0.95}},
  {{"term":"entity linking",    "weight":0.80}},
  {{"term":"question answering","weight":0.70}},
  {{"term":"recommendation systems","weight":0.65}},
  {{"term":"semantic context",  "weight":0.60}}
]}}

Example 3:
<TEXT>
Graph neural networks (GNNs) extend deep learning to non-Euclidean graph data by iteratively aggregating neighborhood information. Popular variants include Graph Convolutional Networks (GCNs), Graph Attention Networks (GATs), and Message Passing Neural Networks (MPNNs).
</TEXT>
Expected output:
{{"concepts":[
  {{"term":"graph neural network",                "weight":0.95}},
  {{"term":"Graph Convolutional Network (GCN)",   "weight":0.85}},
  {{"term":"Graph Attention Network (GAT)",       "weight":0.85}},
  {{"term":"Message Passing Neural Network (MPNN)","weight":0.80}},
  {{"term":"neighborhood aggregation",            "weight":0.75}},
  {{"term":"node classification",                 "weight":0.65}},
  {{"term":"link prediction",                     "weight":0.60}},
  {{"term":"molecular property prediction",       "weight":0.55}}
]}}

Example 4:
<TEXT>
To speed up query performance, modern database systems often employ B-tree and LSM-tree indexes …
</TEXT>
Expected output:
{{"concepts":[
  {{"term":"B-tree index",          "weight":0.90}},
  {{"term":"LSM-tree index",        "weight":0.90}},
  {{"term":"logarithmic search time","weight":0.75}},
  {{"term":"write buffering",       "weight":0.70}},
  {{"term":"batch disk writes",     "weight":0.70}},
  {{"term":"secondary index",       "weight":0.65}},
  {{"term":"inverted list",         "weight":0.60}},
  {{"term":"hash index",            "weight":0.60}}
]}}

Example 5:
<TEXT>
In distributed consensus, Raft and Paxos are two foundational algorithms …
</TEXT>
Expected output:
{{"concepts":[
  {{"term":"distributed consensus", "weight":0.95}},
  {{"term":"Raft algorithm",        "weight":0.90}},
  {{"term":"leader election",       "weight":0.85}},
  {{"term":"log replication",       "weight":0.80}},
  {{"term":"Paxos algorithm",       "weight":0.80}},
  {{"term":"gossip protocol",       "weight":0.65}},
  {{"term":"vector clock",          "weight":0.60}}
]}}

---
Now, without repeating the above examples, extract concepts for the following paragraph:
<TEXT>
{paragraph}
</TEXT>
Expected output (JSON ponly, no extra test):
"""
    
# This is the generation pipeline 
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


def extract_concepts(paragraph: str, debug=False):
    prompt = PROMPT_TEMPLATE.format(paragraph=paragraph.strip())
    response = generator(prompt, max_new_tokens=400,
                         temperature=0.0, do_sample=False)[0]["generated_text"]

    if debug:
        print("RAW:\n", response, "\n---")

    for chunk in re.findall(r"\{[^{}]+\}", response, re.S)[::-1]:
        try:
            data = json.loads(chunk)
            if "concepts" in data:
                return data["concepts"]
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON with 'concepts' found.")



import os, re, json, warnings, pandas as pd, torch
from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
warnings.filterwarnings("ignore", category=UserWarning)

# This is neo4j driver (calling neo4j)
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI",  "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"),
          os.getenv("NEO4J_PASS", "Manami1008"))
)

# UNWIND allows match each concept separately and then group back by paper to sum up weights
# And then check all papers by comparing title, abstract, topic, or field of study.
# After it checks t, next it checks different term and repeat
# Its better to add all the weights and shows as relevance. For example, if the paper uses the word "Recemmender system", "LLM", the paper which includes both term would be higher. 
# Recommend by order of the relevance 
CYPHER_WEIGHTS = """
UNWIND $terms AS t
MATCH (p:Paper)
WHERE toLower(p.title)    CONTAINS t
   OR toLower(p.abstract) CONTAINS t
   OR EXISTS {
         MATCH (p)-[:HAS_TOPIC|:HAS_FOS]->(x)
         WHERE toLower(x.name) CONTAINS t }
WITH p, t
RETURN p.id   AS id,
       p.title AS title,
       p.year  AS year,
       sum($weights[t]) AS relevance
ORDER BY relevance DESC, year DESC
LIMIT 50
"""

def query_with_weights(kws):
    terms   = [k["term"].lower() for k in kws]
    weights = {k["term"].lower(): k["weight"] for k in kws}
    with driver.session() as s:
        rows = s.run(CYPHER_WEIGHTS, terms=terms, weights=weights).data()
    return pd.DataFrame(rows)

# demo
if __name__ == "__main__":
    paragraph = (
          "Recently, as advanced natural language processing techniques, Large Language Models (LLMs) with billion parameters have generated large impacts on various research fields such as Natural Language Processing (NLP), Computer Vision, and Molecule Discovery. Technically most existing LLMs are transformer-based models pre-trained on a vast amount of textual data from diverse sources, such as articles, books, websites, and other publicly available written materials. As the parameter size of LLMs continues to scale up with a larger training corpus, recent studies indicated that LLMs can lead to the emergence of remarkable capabilities. More specifically, LLMs have demonstrated the unprecedentedly powerful abilities of their fundamental responsibilities in language understanding and generation. These improvements enable LLMs to better comprehend human intentions and generate language responses that are more human-like in nature. Moreover, recent studies indicated that LLMs exhibit impressive generalization and reasoning capabilities, making LLMs better generalize to a variety of unseen tasks and domains. To be specific, instead of requiring extensive fine-tuning on each specific task, LLMs can apply their learned knowledge and reasoning skills to fit new tasks simply by providing appropriate instructions or a few task demonstrations. Advanced techniques such as in-context learning can further enhance such generalization performance of LLMs without being fine-tuned on specific downstream tasks. In addition, empowered by prompting strategies such as chain-of-thought, LLMs can generate the outputs with step-by-step reasoning in complicated decision-making processes.Hence, given their powerful abilities, LLMs demonstrate great potential to revolutionize recommender systems."
    )
    
    kw_list = extract_concepts(paragraph, debug=True)
    print("Weighted concepts:", kw_list)
    df = query_with_weights(kw_list)
    print(df.head(10).to_string(index=False))
