# This is direct search using neo4j without using LLM
from neo4j import GraphDatabase
import os, pandas as pd

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI",  "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"),
          os.getenv("NEO4J_PASS", "Manami1008"))  
)

# === Cell: helper to query papers by concept list ===
BASE_CYPHER = """
MATCH (p:Paper)
WHERE
  // title / abstract lexical hit
  ANY(t IN $terms WHERE toLower(p.title)    CONTAINS t) OR
  ANY(t IN $terms WHERE toLower(p.abstract) CONTAINS t) OR
  // topic / FoS match
  EXISTS {
     MATCH (p)-[:HAS_TOPIC|:HAS_FOS]->(x)
     WHERE ANY(t IN $terms WHERE toLower(x.name) CONTAINS t)
  }
RETURN p.id AS id,
       p.title AS title,
       p.year  AS year
ORDER BY p.year DESC
LIMIT 50
"""

def graph_search(concepts):
    """concepts: list[str]  ->  pandas.DataFrame (id,title,year)"""
    # keep multi-word phrases & normalise
    terms = [c.lower() for c in concepts if len(c.split()) >= 2]
    if not terms:
        return pd.DataFrame(columns=["id","title","year"])
    with driver.session() as s:
        rows = s.run(BASE_CYPHER, terms=terms).data()
    return pd.DataFrame(rows)

# === Cell: end-to-end demo ===
paragraph = """
Recently, as advanced natural language processing techniques, Large Language Models (LLMs) with billion parameters have generated large impacts on various research fields such as Natural Language Processing (NLP), Computer Vision, and Molecule Discovery. Technically most existing LLMs are transformer-based models pre-trained on a vast amount of textual data from diverse sources, such as articles, books, websites, and other publicly available written materials. As the parameter size of LLMs continues to scale up with a larger training corpus, recent studies indicated that LLMs can lead to the emergence of remarkable capabilities. More specifically, LLMs have demonstrated the unprecedentedly powerful abilities of their fundamental responsibilities in language understanding and generation. These improvements enable LLMs to better comprehend human intentions and generate language responses that are more human-like in nature. Moreover, recent studies indicated that LLMs exhibit impressive generalization and reasoning capabilities, making LLMs better generalize to a variety of unseen tasks and domains. To be specific, instead of requiring extensive fine-tuning on each specific task, LLMs can apply their learned knowledge and reasoning skills to fit new tasks simply by providing appropriate instructions or a few task demonstrations. Advanced techniques such as in-context learning can further enhance such generalization performance of LLMs without being fine-tuned on specific downstream tasks. In addition, empowered by prompting strategies such as chain-of-thought, LLMs can generate the outputs with step-by-step reasoning in complicated decision-making processes.Hence, given their powerful abilities, LLMs demonstrate great potential to revolutionize recommender systems.
"""
concepts = extract_concepts(paragraph)
print("Extracted concepts:", concepts)

df = graph_search(concepts)
print(f"\n⬇  {len(df)} candidate papers")
display(df.head(10))
