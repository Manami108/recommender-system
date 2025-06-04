# paragraph_kg.py
# ──────────────────────────────────────────────────────────────
# Build a mini–knowledge graph from a text paragraph:
#   1. Extract entities via spaCy NER
#   2. Extract relational triples via LLaMA prompting
#   3. Construct an in-memory graph (networkx) and optionally push to Neo4j
# ──────────────────────────────────────────────────────────────

import re, json, networkx as nx, spacy
from typing import List, Tuple
from neo4j import GraphDatabase

# ------ 1. lightweight NER for node candidates ----------------
nlp = spacy.load("en_core_web_sm")          # ≈ 14 MB, fast CPU runtime

def extract_entities(paragraph: str) -> List[str]:
    """Return unique entity surface strings (order preserved)."""
    doc   = nlp(paragraph)
    keep  = {"ORG", "PERSON", "GPE", "NORP", "PRODUCT", "EVENT", "WORK_OF_ART"}
    ents  = [ent.text for ent in doc.ents if ent.label_ in keep]
    # simple dedupe while preserving order
    seen, unique = set(), []
    for e in ents:
        if e not in seen:
            seen.add(e)
            unique.append(e)
    return unique


# ------ 2. relation extraction with your LLaMA generator -------
TRIPLE_PROMPT = """
You are an expert relation extractor for computer-science research text.

**Task**  
From the text enclosed by <TEXT></TEXT>  
 • Extract **up to 20** factual triples in the exact JSON format\n
   ```json
   [["head","relation","tail"], …]
   ```\n
 • Use concise relation labels (e.g. "uses", "extends", "improves").  
 • Heads/tails should be noun phrases that appear verbatim in the text.  
 • Output **only** the JSON list (no commentary).

––––– Examples –––––

Example 1  
<TEXT>
Knowledge graphs represent entities and relations as structured graphs.  
They are widely used for tasks like entity linking, question answering, and recommendation systems.
</TEXT>
Expected:
[["Knowledge graphs","represent","entities and relations"],
 ["Knowledge graphs","used_for","entity linking"],
 ["Knowledge graphs","used_for","question answering"],
 ["Knowledge graphs","used_for","recommendation systems"]]

Example 2  
<TEXT>
Graph Neural Networks (GNNs) extend deep learning to non-Euclidean data by aggregating neighborhood information.  
Popular variants include Graph Convolutional Networks (GCNs) and Graph Attention Networks (GATs).
</TEXT>
Expected:
[["Graph Neural Networks","extend","deep learning"],
 ["Graph Neural Networks","aggregate","neighborhood information"],
 ["Graph Convolutional Networks","variant_of","Graph Neural Networks"],
 ["Graph Attention Networks","variant_of","Graph Neural Networks"]]

Example 3  
<TEXT>
B-tree indexes support balanced, ordered data access with logarithmic search time,  
whereas LSM-trees buffer writes in memory and flush them to disk for high throughput.
</TEXT>
Expected:
[["B-tree indexes","provide","balanced ordered access"],
 ["B-tree indexes","achieve","logarithmic search time"],
 ["LSM-trees","buffer","writes in memory"],
 ["LSM-trees","flush","writes to disk"],
 ["LSM-trees","provide","high write throughput"]]

Example 4  
<TEXT>
The Raft algorithm divides consensus into leader election, log replication, and safety,  
while Paxos relies on proposer, acceptor, and learner roles.
</TEXT>
Expected:
[["Raft algorithm","divides_into","leader election"],
 ["Raft algorithm","divides_into","log replication"],
 ["Raft algorithm","divides_into","safety"],
 ["Paxos","relies_on","proposer role"],
 ["Paxos","relies_on","acceptor role"],
 ["Paxos","relies_on","learner role"]]

Example 5  
<TEXT>
Transformer-based Large Language Models (LLMs) are pre-trained on massive text corpora  
and can solve new tasks by in-context learning without gradient updates.
</TEXT>
Expected:
[["Transformer-based Large Language Models","pre_trained_on","massive text corpora"],
 ["Transformer-based Large Language Models","solve","new tasks"],
 ["Transformer-based Large Language Models","use","in-context learning"],
 ["in-context learning","requires","no gradient updates"]]

––––– Your turn –––––

Now extract triples for the following paragraph. Do not limit to the words' examples provided. 
Think by yourself based on the logic provided. 
Especially edges, it does not have to be like example, but something verb which is wrriten or can connect noun.
Return **only** the JSON list.
Important: Output ONLY the JSON list. Do NOT add code fences, back-ticks, or commentary.

<TEXT>
{paragraph}
</TEXT>
"""

import re, json, ast

def _first_json_list_after(text: str, anchor: str = "</TEXT>") -> str:
    # Get everything after </TEXT>
    after = text.split(anchor, 1)[-1]
    # Find all bracketed lists
    blocks = re.findall(r"\[[\s\S]*?\]", after)
    # Return the longest one (most likely the real output)
    return max(blocks, key=len) if blocks else None

def _sanitize(block: str) -> str:
    # Remove backtick fences and whitespace
    blk = block.replace("```json", "").replace("```", "").strip()
    # Normalize quotes
    blk = blk.replace("“", '"').replace("”", '"')
    blk = re.sub(r"(?<!\\)'", '"', blk)
    # Drop trailing commas before the closing bracket
    blk = re.sub(r",\s*\]", "]", blk)
    return blk
def _balance_brackets(s: str) -> str:
    open_count  = s.count('[')
    close_count = s.count(']')
    if open_count > close_count:
        s += ']' * (open_count - close_count)
    return s

def _parse_triples(block: str):
    blk = _sanitize(block)
    blk = _balance_brackets(blk)    # ← auto-close any unbalanced lists
    for loader in (json.loads, ast.literal_eval):
        try:
            data = loader(blk)
            return [tuple(x) for x in data if len(x)==3]
        except:
            continue
    return None


def extract_triples(paragraph: str, generator) -> list[tuple[str,str,str]]:
    prompt = TRIPLE_PROMPT.format(paragraph=paragraph)
    out    = generator(prompt,
                      max_new_tokens=800,
                      temperature=0.0,
                      do_sample=False)[0]["generated_text"]

    block = _first_json_list_after(out)
    if not block:
        print("No JSON block found after </TEXT>")
        return []

    triples = _parse_triples(block)
    if triples is None:
        print("Still could not parse block:\n", block[:200])
        return []
    return triples

# ------ 3. build the in-memory graph ---------------------------
def build_paragraph_kg(paragraph: str, generator) -> nx.MultiDiGraph:
    """Return a networkx MultiDiGraph representing the paragraph KG."""
    ents    = extract_entities(paragraph)
    triples = extract_triples(paragraph, generator)

    G = nx.MultiDiGraph()

    # add entity nodes
    for e in ents:
        G.add_node(e, type="entity")

    # add triples
    for h, r, t in triples:
        for node in (h, t):
            if node not in G:
                G.add_node(node, type="entity")
        G.add_edge(h, t, label=r)

    return G




# ------ 4. optional: push to Neo4j -----------------------------
CREATE_NODE = """
MERGE (c:ParaConcept {name:$name})
RETURN id(c) AS id
"""
CREATE_EDGE = """
MATCH (h:ParaConcept {name:$h}),
      (t:ParaConcept {name:$t})
MERGE (h)-[:PARA_REL {type:$rel}]->(t)
"""

def push_to_neo4j(G: nx.MultiDiGraph, driver: GraphDatabase.driver):
    with driver.session() as session:
        # nodes
        for n in G.nodes:
            session.run(CREATE_NODE, name=n)
        # edges
        for h, t, data in G.edges(data=True):
            session.run(CREATE_EDGE, h=h, t=t, rel=data.get("label", ""))


# -------------------------- demo -------------------------------
if __name__ == "__main__":
    from hybrid_search import generator, driver  # re-use your objects

    test_para = "Recently, as advanced natural language processing techniques, Large Language Models (LLMs) with billion parameters have generated large impacts on various research fields such as Natural Language Processing (NLP), Computer Vision, and Molecule Discovery. Technically most existing LLMs are transformer-based models pre-trained on a vast amount of textual data from diverse sources, such as articles, books, websites, and other publicly available written materials. As the parameter size of LLMs continues to scale up with a larger training corpus, recent studies indicated that LLMs can lead to the emergence of remarkable capabilities. More specifically, LLMs have demonstrated the unprecedentedly powerful abilities of their fundamental responsibilities in language understanding and generation. These improvements enable LLMs to better comprehend human intentions and generate language responses that are more human-like in nature. Moreover, recent studies indicated that LLMs exhibit impressive generalization and reasoning capabilities, making LLMs better generalize to a variety of unseen tasks and domains. To be specific, instead of requiring extensive fine-tuning on each specific task, LLMs can apply their learned knowledge and reasoning skills to fit new tasks simply by providing appropriate instructions or a few task demonstrations. Advanced techniques such as in-context learning can further enhance such generalization performance of LLMs without being fine-tuned on specific downstream tasks. In addition, empowered by prompting strategies such as chain-of-thought, LLMs can generate the outputs with step-by-step reasoning in complicated decision-making processes.Hence, given their powerful abilities, LLMs demonstrate great potential to revolutionize recommender systems."
    kg = build_paragraph_kg(test_para, generator)
    print("Nodes:", kg.nodes(data=True))
    print("Edges:", list(kg.edges(data=True)))

    # Uncomment to persist in Neo4j
    # push_to_neo4j(kg, driver)
