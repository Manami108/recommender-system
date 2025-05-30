# This is only hybrid search based recommendation 
# But still, need to think about how to make it faster because now its running on cpu
# Need to think about importance score for the keywords matching 
# Need to think about path length. hops reasoning 

import os
import re
import json
import warnings
import pandas as pd
import torch

from neo4j import GraphDatabase
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
print("▶ Loading LLaMA 3-8B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16, # I think this is memmory efficient, less VRAM usage!!
    device_map="auto" #This line is to make sure model is loaded on GPU (however, in Manami's computer, it becomes CPU cuz simply, the my gpu cannot handle it. )
)

# This is the generation pipeline 
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Extract concepts with refined few-shot prompting
def extract_concepts(paragraph: str):
    prompt = f"""
You are an academic assistant of computer science field. Extract the most important research concepts (keywords of the from the paragraph wrapped in <TEXT> tags.
Respond only with a JSON object of the form {{"concepts": ["concept1","concept2",…]}}. Extract some unique terms rather than common words in computer science paper paragraph. 

Example 1:
<TEXT>
Transformer-based architectures, like BERT and GPT, have revolutionized NLP by enabling bidirectional attention and large-scale pretraining.
These models achieve state-of-the-art results in tasks such as question answering, machine translation, and text summarization.
</TEXT>
Expected output:
{{"concepts": ["transformer", "BERT", "GPT", "bidirectional attention", "pretraining", "question answering", "machine translation", "text summarization"]}}

Example 2:
<TEXT>
Knowledge graphs represent entities and their relations as a structured graph. They are widely used in tasks like entity linking, question answering, and recommendation systems.
</TEXT>
Expected output:
{{"concepts": ["knowledge graph","entity linking", "question answering", "recommendation systems", "semantic context"]}}

Example 3:
<TEXT>
Graph neural networks (GNNs) extend deep learning to non-Euclidean graph data by iteratively aggregating neighborhood information.  
Popular variants include Graph Convolutional Networks (GCNs), Graph Attention Networks (GATs), and Message Passing Neural Networks (MPNNs).  
They’ve been applied to node classification, link prediction, and molecular property prediction.
</TEXT>
Expected output:
{{"concepts": ["graph neural network", "neighborhood aggregation", "Graph Convolutional Network (GCN)", "Graph Attention Network (GAT)", "Message Passing Neural Network (MPNN)", "node classification", "link prediction", "molecular property prediction"]}}

Example 4:
<TEXT>
To speed up query performance, modern database systems often employ B-tree and LSM-tree indexes.  
B-trees support balanced, ordered data access with logarithmic search time, while Log-Structured Merge trees buffer writes in memory and batch them to disk for high write throughput.  
Secondary indexes like inverted lists or hash indexes accelerate lookups on non-primary key columns.
</TEXT>
Expected output:
{{"concepts": ["B-tree index", "LSM-tree index", "logarithmic search time", "write buffering", "batch disk writes", "secondary index", "inverted list", "hash index", "non-primary key lookup"]}}

Example 5:
<TEXT>
In distributed consensus, Raft and Paxos are two foundational algorithms.  
Raft divides the problem into leader election, log replication, and safety, making it more understandable.  
Paxos focuses on proposer, acceptor, and learner roles to reach agreement despite failures.  
Gossip protocols and vector clock mechanisms are also widely used for state propagation and causality tracking.
</TEXT>
Expected output:
{{"concepts": ["distributed consensus", "Raft algorithm", "leader election", "log replication", "Paxos algorithm", "proposer role", "acceptor role", "learner role", "gossip protocol", "vector clock"]}}


---
Now, without repeating the above examples, extract concepts for the following paragraph:
<TEXT>
{paragraph}
</TEXT>
Expected output (JSON only, no extra text):

"""

    result = generator(
        prompt,
        max_new_tokens=600,
        temperature=0.0,
        do_sample=False
    )[0]["generated_text"]

# Max new tokens are set to 600 to limit the LLM's answer (but the token is in addition to the input)
# temperature tells what kind of output. For example, 0.0 tells always same output for the same input but 1.0 has balanced randomness. so 0.0 tells conssistent and accurate answers
# Sample also tells the randomness. so if i set sample = True, then temerature tells how much randomness i want.


    # Extract valid JSON from output
    # This line try to search all strings like this (json-looking substrings) and (re is search for patterns)
    # The json file is made like this {"concepts": ["knowledge graphs", "AI systems", "data integration"]}
    matches = re.findall(r"\{[^{}]+\}", result, re.S)
    if matches:
        last = matches[-1]
        try:
            data = json.loads(last)
            if "concepts" in data and isinstance(data["concepts"], list):
                return data["concepts"]
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse model output for paragraph. Full output:\n{result}")




# SciBERT embedding 
print("▶ Loading SciBERT embedder…")
sci_model = SentenceTransformer("allenai/scibert_scivocab_uncased")
sci_model.eval()

def embed(text: str) -> list[float]:
    # returns normalized embedding
    vec = sci_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return vec.tolist()

def vector_search(qvec, top_k=25) -> pd.DataFrame:
    with driver.session() as s:
        rows = s.run(
            """
            CALL db.index.vector.queryNodes('paper_vec', $k, $vec)
            YIELD node, score
            RETURN node.id AS id, 1.0 - score AS sim
            ORDER BY score ASC
            """, k=top_k, vec=qvec
        ).data()
    return pd.DataFrame(rows)


def hydrate_paper_meta(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    ids = df["id"].tolist()
    with driver.session() as s:
        meta = s.run(
            "MATCH (p:Paper) WHERE p.id IN $ids RETURN p.id AS id, p.title AS title, p.year AS year",
            ids=ids
        ).data()
    return df.merge(pd.DataFrame(meta), on="id", how="left")[["id","title","year","sim"]]


# This is neo4j driver (calling neo4j)
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI",  "bolt://localhost:7687"),
    auth=(
        os.getenv("NEO4J_USER","neo4j"),
        os.getenv("NEO4J_PASS","Manami1008")
    )
)

# MATCH (p:Paper) selects all paper labels, and WHERE tells the paper which matches with concepts etraxted from user's paragraph.
# $terms is a parameter which is passed into
# So its like "Is there any term t in the input $terms list such that lowercase paper title contains that term?"
# The keywords are seached in title, abstracts, and field of study or topics. 
# But for as topics and as field of study (x), it checks one hop to search paper p
# Return paper id, title, year (now, its sorted by year cuz there is no ranking method)

BASE_CYPHER = """
MATCH (p:Paper)
WHERE 
  ANY(t IN $terms WHERE toLower(p.title)    CONTAINS t) OR
  ANY(t IN $terms WHERE toLower(p.abstract) CONTAINS t) OR
  EXISTS {
    MATCH (p)-[:HAS_TOPIC|:HAS_FOS]->(x)
    WHERE ANY(t IN $terms WHERE toLower(x.name) CONTAINS t)
  }
RETURN p.id AS id, p.title AS title, p.year AS year
ORDER BY p.year DESC
"""

# The concept in this is accept a list of concept keywords extracted from a paragraph
# And run Cypher query agianst Neo4j knowledge graph
# Return a paper
# It returns a pandas.DataFrame object containing search results from Neo4j.
# Filters out single word tems by keeping 2 o more words to reduce noise. Multi-word is always better no?
# After filltering it out returns dataFrame
# rows = s.run(BASE_CYPHER, terms=terms).data() this executes the BASE_CYPHER query and the pass the terms list into $terms


def graph_search(concepts: list[str]) -> pd.DataFrame:
    terms = [c.lower() for c in concepts if len(c.split()) >= 2]
    if not terms:
        return pd.DataFrame(columns=["id","title","year"])
    with driver.session() as s:
        rows = s.run(BASE_CYPHER, terms=terms).data()
    return pd.DataFrame(rows)


# Demo
if __name__ == "__main__":
    paragraph = (
       "Recently, as advanced natural language processing techniques, Large Language Models (LLMs) with billion parameters have generated large impacts on various research fields such as Natural Language Processing (NLP), Computer Vision, and Molecule Discovery. Technically most existing LLMs are transformer-based models pre-trained on a vast amount of textual data from diverse sources, such as articles, books, websites, and other publicly available written materials. As the parameter size of LLMs continues to scale up with a larger training corpus, recent studies indicated that LLMs can lead to the emergence of remarkable capabilities. More specifically, LLMs have demonstrated the unprecedentedly powerful abilities of their fundamental responsibilities in language understanding and generation. These improvements enable LLMs to better comprehend human intentions and generate language responses that are more human-like in nature. Moreover, recent studies indicated that LLMs exhibit impressive generalization and reasoning capabilities, making LLMs better generalize to a variety of unseen tasks and domains. To be specific, instead of requiring extensive fine-tuning on each specific task, LLMs can apply their learned knowledge and reasoning skills to fit new tasks simply by providing appropriate instructions or a few task demonstrations. Advanced techniques such as in-context learning can further enhance such generalization performance of LLMs without being fine-tuned on specific downstream tasks. In addition, empowered by prompting strategies such as chain-of-thought, LLMs can generate the outputs with step-by-step reasoning in complicated decision-making processes.Hence, given their powerful abilities, LLMs demonstrate great potential to revolutionize recommender systems."
    )

    concepts = extract_concepts(paragraph)
    print("\n▶ Extracted concepts:\n", concepts)

    df_graph = graph_search(concepts)
    print(f"\n⬇ Graph search ({len(df_graph)} hits):")
    print(df_graph.head(25).to_string(index=False))

    df_vec = vector_search(embed(paragraph), top_k=25)
    df_vec = hydrate_paper_meta(df_vec)
    print(f"\n⬇ Vector search ({len(df_vec)} hits):")
    print(df_vec.head(25).to_string(index=False, formatters={"sim":"{:.3f}".format}))
