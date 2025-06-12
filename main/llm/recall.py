# recall.py
# ---------------------------------------------------------------------------
# Retrieval utilities for the context-aware scientific-paper recommender
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Iterable

import numpy as np
import pandas as pd
from neo4j import GraphDatabase, READ_ACCESS
from sentence_transformers import SentenceTransformer

# ────────────────────────────── CONFIGURATION ────────────────────────────── #

# Neo4j
_NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
_NEO4J_PASS = os.getenv("NEO4J_PASS", "Manami1008")

# Vector & BM25 indexes (names must match those created in Neo4j)
VECTOR_INDEX   = "paper_vec"
FULLTEXT_INDEX = "paper_fulltext"

# Search hyper-parameters (override per-call if needed)
DEFAULT_K_BM25   = 40
DEFAULT_K_VEC    = 40
DEFAULT_SIM_TH   = 0.30         # cosine similarity threshold (1–distance)
EMBED_DIM        = 768          # SciBERT size; used for zero-vec fallback

# ─────────────────────────────── CONNECTIONS ─────────────────────────────── #

# A single global driver is fine for most scripts.  Neo4j handles pooling.
_driver = GraphDatabase.driver(
    _NEO4J_URI,
    auth=(_NEO4J_USER, _NEO4J_PASS),
)

# ──────────────────────────── EMBEDDING / ENCODING ───────────────────────── #

@lru_cache(maxsize=1)
def _sci_model() -> SentenceTransformer:
    """Singleton-style loader so the SciBERT weights load only once."""
    model = SentenceTransformer("allenai/scibert_scivocab_uncased")
    model.eval()
    return model

def embed(text: str) -> np.ndarray:
    """Return a *normalized* 768-d SciBERT embedding (L2 = 1)."""
    return _sci_model().encode(text, convert_to_numpy=True, normalize_embeddings=True)

# ────────────────────────────── LUCENE ESCAPING ──────────────────────────── #

_LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]^"~*?:\\/])')

def escape_lucene(text: str, *, max_len: int = 500) -> str:
    """Escape Lucene special chars and truncate to ‹max_len› (index limit)."""
    return _LUCENE_SPECIAL.sub(r"\\\1", text[:max_len])

# ────────────────────────────── CORE RETRIEVAL ───────────────────────────── #

def recall_fulltext(
    query: str,
    *,
    k: int = DEFAULT_K_BM25,
) -> pd.DataFrame:
    """BM25 lookup over title+abstract FULLTEXT index."""
    lucene_q = escape_lucene(query)
    cypher = """
        CALL db.index.fulltext.queryNodes($idx, $q, {limit:$k})
        YIELD node, score
        RETURN node.id AS pid,
               node.title AS title,
               node.year  AS year,
               0          AS hop,
               score      AS sim
    """
    with _driver.session(default_access_mode=READ_ACCESS) as sess:
        rows = sess.run(cypher, idx=FULLTEXT_INDEX, q=lucene_q, k=k).data()
    return pd.DataFrame(rows, columns=["pid", "title", "year", "hop", "sim"])


def recall_vector(
    vec: np.ndarray,
    *,
    k: int = DEFAULT_K_VEC,
    sim_threshold: float = DEFAULT_SIM_TH,
) -> pd.DataFrame:
    """Approximate nearest-neighbour search on the `paper_vec` index."""
    cypher = """
        CALL db.index.vector.queryNodes($idx, $k, $vec)
        YIELD node, score
        WITH node, (1.0 - score) AS sim
        WHERE sim >= $th
        RETURN node.id    AS pid,
               node.title AS title,
               node.year  AS year,
               0          AS hop,
               sim
    """
    with _driver.session(default_access_mode=READ_ACCESS) as sess:
        rows = sess.run(
            cypher,
            idx=VECTOR_INDEX,
            k=k,
            vec=vec.astype(np.float32).tolist(),
            th=float(sim_threshold),
        ).data()
    return pd.DataFrame(rows, columns=["pid", "title", "year", "hop", "sim"])


def recall_by_chunks(
    chunks: Iterable[str],
    *,
    k_bm25: int = DEFAULT_K_BM25,
    k_vec: int  = DEFAULT_K_VEC,
    sim_th: float = DEFAULT_SIM_TH,
) -> pd.DataFrame:
    """
    For every *chunk* of the query paragraph, run both BM25 and vector recall.
    Returns one row per (pid, source) with the best similarity score.
    """
    rows: list[dict[str, Any]] = []

    for ch in chunks:
        rows += recall_fulltext(ch, k=k_bm25)           .assign(source="bm25").to_dict("records")
        rows += recall_vector(embed(ch), k=k_vec,
                              sim_threshold=sim_th)      .assign(source="embed").to_dict("records")

    if not rows:
        return pd.DataFrame(columns=["pid", "title", "year", "hop", "sim", "source"])

    df = (
        pd.DataFrame(rows)
          .sort_values(["source", "sim"], ascending=[True, False])
          .drop_duplicates(["pid", "source"], keep="first")   # keep best hit per source
          .reset_index(drop=True)
    )
    return df

# ────────────────────────────── METADATA LOOKUP ───────────────────────────── #

def fetch_metadata(pids: list[str]) -> pd.DataFrame:
    """
    Return ‹pid, title, abstract, authors, year› for each paper id in ‹pids›.
    Always yields the same columns (possibly empty) to avoid KeyErrors
    downstream.
    """
    if not pids:
        return pd.DataFrame(columns=["pid", "title", "abstract", "authors", "year"])

    cypher = """
        MATCH (p:Paper)
        WHERE p.id IN $ids
        RETURN p.id       AS pid,
               p.title    AS title,
               COALESCE(p.abstract, '') AS abstract,
               COALESCE(p.authors,  '') AS authors,
               p.year     AS year
    """
    with _driver.session(default_access_mode=READ_ACCESS) as sess:
        rows = sess.run(cypher, ids=pids).data()

    # Ensure DataFrame has all expected columns even if Neo4j returns 0 rows
    return pd.DataFrame(rows, columns=["pid", "title", "abstract",
                                       "authors", "year"])

# ──────────────────────────── COSINE SIM UTILITY ─────────────────────────── #

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Assumes vectors are already L2-normalized → dot-product = cosine."""
    return float(np.dot(a, b))

# ───────────────────────────── PUBLIC RE-EXPORTS ─────────────────────────── #

__all__ = [
    # embedding & utils
    "embed", "cosine_similarity",
    # recall
    "recall_fulltext", "recall_vector", "recall_by_chunks",
    # metadata
    "fetch_metadata",
]
