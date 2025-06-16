# This code is to recall potentially relevant papers using BM25 and vector similarity search 

from __future__ import annotations
import os
# Suppress tqdm progress bars
os.environ["TRANSFORMERS_NO_TQDM"] = "true"
os.environ["SENTENCE_TRANSFORMERS_NO_TQDM"] = "true"

import re
from functools import lru_cache
from typing import List, Iterable

import numpy as np
import pandas as pd
from neo4j import GraphDatabase, READ_ACCESS
from sentence_transformers import SentenceTransformer

# config

# Neo4j
_NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
_NEO4J_PASS = os.getenv("NEO4J_PASS", "Manami1008")

# This was created first, but please check neo4j command 
VECTOR_INDEX   = "paper_vec"
FULLTEXT_INDEX = "paper_fulltext"

# default hyper parameters
defaults = {
    "k_full_bm25": 20,
    "k_full_vec": 20,
    "k_chunk_bm25": 10,
    "k_chunk_vec": 10,
    "sim_threshold": 0.30,
    "hybrid_w_lex": 0.5,
    "hybrid_w_sem": 0.5,
}      

# neo4j driver 

dr = GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASS))

# SciBERT loading 
@lru_cache(maxsize=1)
def _sci_model() -> SentenceTransformer:
    """Singleton loader without progress bars"""
    model = SentenceTransformer("allenai/scibert_scivocab_uncased")
    model.eval()
    return model


def embed(text: str) -> np.ndarray:
    vec = _sci_model().encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return vec

# This is a safety limit to truncate very long input text before passing it into the Lucene full text search query to neo4j. 
# Very only query (more than 500) can crash parser. 
# BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document, regardless of their proximity within the document. 
_LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]^"~*?:\\/])')

def escape_lucene(text: str, *, max_len: int = 500) -> str:
    return _LUCENE_SPECIAL.sub(r"\\\\1", text[:max_len])

# Input is string query (user paragraph in this case), and output is dataframe of top-k papers by BM25 relevance.
# papers published in 2020 is removed (because testset is 2020 papers)
def recall_fulltext(
    query: str,
    k: int = defaults["k_full_bm25"]
) -> pd.DataFrame:
    lucene_q = escape_lucene(query)
    cypher = '''
        CALL db.index.fulltext.queryNodes($idx, $q, {limit:$k})
        YIELD node, score
        WHERE node.year <> 2020
        RETURN node.id AS pid, score AS bm25_score
    '''
    with dr.session(default_access_mode=READ_ACCESS) as sess:
        rows = sess.run(cypher, idx=FULLTEXT_INDEX, q=lucene_q, k=k).data()
    df = pd.DataFrame(rows)
    df = df.assign(
        source='full_bm25',
        rank=np.arange(1, len(df)+1)
    )
    return df


# In this block, input is embedding vector and output is dataframe of similar papers using vector similarity search. 
# But the default similarity is set to 0.3, and if it is not above threthold, return zero maybe. 
# So again, it filters out low-similarity results and test year.
def recall_vector(
    query_vec: np.ndarray,
    k: int = defaults["k_full_vec"],
    sim_th: float = defaults["sim_threshold"]
) -> pd.DataFrame:
    cypher = """
        CALL db.index.vector.queryNodes($idx,$k,$vec)
        YIELD node, score                      // Neo4j always calls this column “score”
        WITH node, (1.0 - score) AS sim        // so the row key will be “sim”
        WHERE sim >= $th AND node.year <> 2020
        RETURN node.id AS pid, sim AS semantic_score
    """
    with dr.session(default_access_mode=READ_ACCESS) as sess:
        rows = sess.run(
            cypher,
            idx=VECTOR_INDEX,
            k=k,
            vec=query_vec.astype(np.float32).tolist(),
            th=sim_th
        ).data()

    # ensure column names are exactly pid + semantic_score
    df = pd.DataFrame(rows, columns=["pid", "semantic_score"])

    df["source"] = "full_vec"
    df["rank"]   = np.arange(1, len(df) + 1)
    return df


# Here, input is paragraph chunks and output is the dataframe of combined BM25 and vector similarity search 
# For now, token-based chunk 
def recall_by_chunks(
    chunks: Iterable[str],
    k_bm25: int = defaults["k_chunk_bm25"],
    k_vec: int = defaults["k_chunk_vec"],
    sim_th: float = defaults["sim_threshold"]
) -> pd.DataFrame:
    records: List[pd.DataFrame] = []
    for chunk in chunks:
        bm = recall_fulltext(chunk, k=k_bm25)
        ve = recall_vector(embed(chunk), k=k_vec, sim_th=sim_th).copy()
        bm["source"] = "chunk_bm25"
        ve["source"] = "chunk_vec"
        records.extend([bm, ve])
    if not records:
        return pd.DataFrame(columns=["pid","bm25_score","semantic_score","source","rank"])
    pool = pd.concat(records, ignore_index=True)
    # ensure semantic_score column exists in BM25-only records
    if "semantic_score" not in pool.columns:
        pool["semantic_score"] = pool.get("semantic_score", 0)
    if "bm25_score" not in pool.columns:
        pool["bm25_score"] = pool.get("bm25_score", 0)
    return pool

# This uses Reciprocal Rank Fusion (RRF) score 
# https://learn.microsoft.com/ja-jp/azure/search/hybrid-search-ranking
# It adds weighted sum of BM25 and semantic on the entire query 
# Chunk: for each paper, take the maximum BM25 / semantic score across all chunks, then form a weighted sum.
# Final: add full-doc and chunk scores for a final hybrid score, then sort.
def hybrid_rank(
    full_bm25: pd.DataFrame,
    full_vec: pd.DataFrame,
    chunk_pool: pd.DataFrame,
    top_k: int = 60,
    w_lex: float = defaults['hybrid_w_lex'],
    w_sem: float = defaults['hybrid_w_sem']
) -> pd.DataFrame:
    # merge full-text and embedding
    full = pd.merge(
        full_bm25[['pid','bm25_score']],
        full_vec[['pid','semantic_score']],
        on='pid', how='outer'
    ).fillna(0)
    # compute hybrid for full doc
    full['hybrid_full'] = w_lex*full['bm25_score'] + w_sem*full['semantic_score']

    # aggregate chunk-level scores: max per pid
    agg = chunk_pool.groupby('pid').agg(
        bm25_chunk=('bm25_score','max'),
        sem_chunk=('semantic_score','max')
    ).reset_index().fillna(0)
    agg['hybrid_chunk'] = w_lex*agg['bm25_chunk'] + w_sem*agg['sem_chunk']

    # join both
    merged = pd.merge(
        full[['pid','hybrid_full']],
        agg[['pid','hybrid_chunk']],
        on='pid', how='outer'
    ).fillna(0)
    merged['hybrid_score'] = merged['hybrid_full'] + merged['hybrid_chunk']

    return merged.sort_values('hybrid_score', ascending=False).head(top_k)


# After that it fetches paper IDs with title, abstract, authors and year. 
# I have to think if abstract is null whats gonna happen. 
# Normally, it does return 0 but then when i do reranking...
def fetch_metadata(pids: List[str]) -> pd.DataFrame:
    if not pids:
        return pd.DataFrame(columns=['pid','title','abstract','authors','year'])
    cypher = '''
        MATCH (p:Paper) WHERE p.id IN $ids
        RETURN p.id AS pid,
               p.title AS title,
               COALESCE(p.abstract, '') AS abstract,
               COALESCE(p.authors, '') AS authors,
               p.year AS year
    '''
    with dr.session(default_access_mode=READ_ACCESS) as sess:
        rows = sess.run(cypher, ids=pids).data()
    return pd.DataFrame(rows)

# ──────────────────────────── COSINE SIM UTILITY ─────────────────────────── #

__all__ = [
    'embed',
    'recall_fulltext', 'recall_vector', 'recall_by_chunks',
    'hybrid_rank', 'fetch_metadata'
]
