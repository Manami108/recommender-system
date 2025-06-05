# kg_driver.py
import os, numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# ── Neo4j driver (load once) ─────────────────────────────
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI",  "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"),
          os.getenv("NEO4J_PASS", "Manami1008"))
)

# ── SciBERT embedder (load once) ─────────────────────────
_sci = SentenceTransformer("allenai/scibert_scivocab_uncased")
_sci.eval()

def embed(txt: str) -> np.ndarray:
    return _sci.encode(txt, convert_to_numpy=True,
                       normalize_embeddings=True)
