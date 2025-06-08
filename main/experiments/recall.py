import os
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# Neo4j connection (singleton)
# URI, USER, PASS via env: NEO4J_URI, NEO4J_USER, NEO4J_PASS
# Default: bolt://localhost:7687, neo4j/neo4j
# ─────────────────────────────────────────────────────────────────────────────
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASS", "Manami1008"))
)

# ─────────────────────────────────────────────────────────────────────────────
# SciBERT embedder (singleton)
# ─────────────────────────────────────────────────────────────────────────────
_sci = SentenceTransformer("allenai/scibert_scivocab_uncased")
_sci.eval()

# ─────────────────────────────────────────────────────────────────────────────
# Index names in Neo4j
# ─────────────────────────────────────────────────────────────────────────────
VECTOR_INDEX   = "paper_vec"
FULLTEXT_INDEX = "paper_fulltext"


def embed(text: str) -> np.ndarray:
    """
    Compute normalized SciBERT embedding for text.
    """
    return _sci.encode(text, convert_to_numpy=True, normalize_embeddings=True)


def recall_fulltext(query: str, k: int = 25) -> pd.DataFrame:
    """
    BM25 recall via full-text index.
    """
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


def recall_vector(vec: np.ndarray, k: int = 25, sim_threshold: float = 0.0) -> pd.DataFrame:
    """
    Vector recall via cosine similarity (1 - Neo4j distance).
    """
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


def recall_by_chunks(chunks: list[str], k_vec: int=40, k_bm25: int=40, sim_th: float=0.3) -> pd.DataFrame:
    """
    For each chunk string, perform BM25 + vector recall and
    aggregate unique candidates keeping best score per pid/src.
    """
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


def expand_citation_hops(pids: list[str], max_hops: int=2, limit_per_hop: int=100) -> pd.DataFrame:
    """
    Breadth-first citation expansion up to 2 hops.
    """
    if not pids or max_hops<1:
        return pd.DataFrame(columns=['pid','title','year','hop'])
    rows = []
    with driver.session() as session:
        # hop1
        rows += session.run(
            """
            UNWIND $pids AS id
            MATCH (p:Paper {id:id})-[:CITES]->(c1:Paper)
            RETURN DISTINCT c1.id AS pid, c1.title AS title,
                   c1.year AS year, 1 AS hop
            LIMIT $lim
            """,
            pids=pids, lim=limit_per_hop
        ).data()
        # hop2
        if max_hops>=2:
            rows += session.run(
                """
                UNWIND $pids AS id
                MATCH (p:Paper {id:id})-[:CITES]->()-[:CITES]->(c2:Paper)
                RETURN DISTINCT c2.id AS pid, c2.title AS title,
                       c2.year AS year, 2 AS hop
                LIMIT $lim
                """,
                pids=pids, lim=limit_per_hop
            ).data()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values('hop').drop_duplicates('pid').reset_index(drop=True)


def fetch_metadata(pids: list[str]) -> pd.DataFrame:
    """
    Fetch title, abstract, authors, year for given pids.
    """
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
