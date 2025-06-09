import os
import numpy as np
import pandas as pd
from neo4j import GraphDatabase # Neo4j
from sentence_transformers import SentenceTransformer # SciBERT
from hop_reasoning import multi_hop_topic_citation_reasoning


# call neo4j driver 
# Need to modify when i want to switch to API
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASS", "Manami1008"))
)

# call sciBERT
_sci = SentenceTransformer("allenai/scibert_scivocab_uncased")
_sci.eval()

# paper vec: vector index stored in neo4j, paper fulltext: full text (BM25) index
VECTOR_INDEX   = "paper_vec"
FULLTEXT_INDEX = "paper_fulltext"

# compute user query embedding, and normalizing is true for cosign similarity computation efficiency. 
def embed(text: str) -> np.ndarray:
    return _sci.encode(text, convert_to_numpy=True, normalize_embeddings=True)

# It retrieves top-k papers which title and abstract matches user query (BM25 so keywords matching based using TFiDF concepts)
# I think 25 is not a good idea because its in the way limit reranking candidates later on. 
# I wrote 0 as hop because it is gonna be exact matching and i will do hop traversal later on. 
def recall_fulltext(query: str, k: int = 25) -> pd.DataFrame:
    rows = []
    with driver.session() as session:
        results = session.run(
            """
            CALL db.index.fulltext.queryNodes($idx, $q, {limit:$k})
            YIELD node, score
            RETURN node.id AS pid, node.title AS title,
                   node.year AS year, 0 AS hop, score AS sim
            """,
            idx=FULLTEXT_INDEX, q=query, k=k
        ).data()
    return pd.DataFrame(results)

# This is embedding search 
# Cosign similarity is computed by (1.0 - score)
# i dont know how much i should set for similarity threshold. 
def recall_vector(vec: np.ndarray, k: int = 25, sim_threshold: float = 0.0) -> pd.DataFrame:
    rows = []
    with driver.session() as session:
        results = session.run(
            """
            CALL db.index.vector.queryNodes($idx, $k, $vec)
            YIELD node, score
            WITH node, (1.0 - score) AS sim
            WHERE sim >= $th
            RETURN node.id AS pid, node.title AS title,
                   node.year AS year, 0 AS hop, sim
            """,
            idx=VECTOR_INDEX, k=k, vec=vec.tolist(), th=sim_threshold
        ).data()
    return pd.DataFrame(results)

# For each chunk, BM25 and embedding are done but i dont know if its good way of doing so. 
def recall_by_chunks(chunks: list[str], k_vec: int=40, k_bm25: int=40, sim_th: float=0.3) -> pd.DataFrame:
    rows = []
    for ch in chunks:
        rows += recall_fulltext(ch, k=k_bm25).assign(src='bm25').to_dict('records')
        vec = embed(ch)
        rows += recall_vector(vec, k=k_vec, sim_threshold=sim_th).assign(src='embed').to_dict('records')

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # keep best sim per pid and src
    df = (
        df.sort_values(['src','sim'], ascending=[True, False])
          .drop_duplicates(['pid','src'])
          .reset_index(drop=True)
    )
    return df

# This is for fetching metadata after all retrieval
def fetch_metadata(pids: list[str]) -> pd.DataFrame:
    if not pids:
        return pd.DataFrame(columns=['pid','title','abstract','authors','year'])
    query = (
        "MATCH (p:Paper) WHERE p.id IN $ids "
        "RETURN p.id AS pid, p.title AS title, p.abstract AS abstract,"
        " p.authors AS authors, p.year AS year"
    )
    with driver.session() as session:
        rows = session.run(query, ids=pids).data()
    return pd.DataFrame(rows)
