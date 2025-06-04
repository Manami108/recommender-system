import os, pandas as pd, numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# ── Neo4j driver ───────────────────────────────────────────
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI",  "bolt://localhost:7687"),
    auth=(
        os.getenv("NEO4J_USER","neo4j"),
        os.getenv("NEO4J_PASS","Manami1008")
    )
)

# ── Text embedder (same SciBERT you used for paper embeddings) ───────────
sci_model = SentenceTransformer("allenai/scibert_scivocab_uncased")
sci_model.eval()

def embed(text: str) -> np.ndarray:
    vec = sci_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return vec

def build_term_lists(ctx: dict) -> dict:
    """
    Take extract_context() output and return:
      • keywords  – list[str] (lower-cased multi-word phrases)
      • tech_terms – list[str] (technologies list from JSON)
      • paragraph_vec – 768-D SciBERT vector
    """
    # 1) collect candidate terms
    kws  = []
    kws += [t.lower() for t in ctx["main_topic"]] if isinstance(ctx["main_topic"], list) else [ctx["main_topic"].lower()]
    kws += [s.lower()  for s in ctx["subtopics"]]
    kws += [t.lower()  for t in ctx["technologies"]]
    
    # keep phrases with ≥2 tokens to avoid noisy 1-word hits
    keywords = [k for k in kws if len(k.split()) >= 2]
    tech_terms = [t for t in ctx["technologies"] if len(t.split()) >= 1]

    # 2) embed full paragraph once (optional semantic search)
    paragraph_vec = embed(" ".join([
        ctx["problem_statement"],
        " ".join(ctx["subtopics"]),
        " ".join(ctx["technologies"])
    ]))
    return {
        "keywords": keywords,
        "tech_terms": tech_terms,
        "paragraph_vec": paragraph_vec
    }

# (A) literal FoS & Topic match
CYPHER_FOS_TOPIC = """
MATCH (p:Paper)-[:HAS_FOS|:HAS_TOPIC]->(x)
WHERE ANY(t IN $terms WHERE toLower(x.name) CONTAINS t)
RETURN p.id AS pid, p.title AS title, p.year AS year,
       1 AS hop, 1.0 AS sim
LIMIT $lim
"""

# (B) paragraph-to-paper vector similarity  (requires 'paper_vec' index)
CYPHER_EMBED = """
CALL db.index.vector.queryNodes('paper_vec', $lim, $vec)
YIELD node, score
RETURN node.id AS pid, node.title AS title, node.year AS year,
       0 AS hop, 1.0 - score AS sim
"""

# (C) one-hop CITES expansion of (A) results
CYPHER_1HOP = """
MATCH (seed:Paper)-[:HAS_FOS|:HAS_TOPIC]->(x)
WHERE ANY(t IN $terms WHERE toLower(x.name) CONTAINS t)
MATCH (seed)-[:CITES]->(p:Paper)
RETURN p.id  AS pid, p.title AS title, p.year AS year,
       2 AS hop, 0.5 AS sim
LIMIT $lim
"""

def retrieve_candidates(ctx: dict,
                        top_n_each: int = 50) -> pd.DataFrame:
    """
    Return a DataFrame with columns: pid, title, year, hop, sim, source
    """
    parts = build_term_lists(ctx)
    terms = parts["keywords"]
    if not terms:
        terms = [w.lower() for w in ctx["main_topic"]]  # fallback
    
    rows = []
    with driver.session() as s:
        # (A) FoS / Topic direct match
        rows += s.run(CYPHER_FOS_TOPIC,
                      terms=terms,
                      lim=top_n_each
                     ).data()
        # flag source
        for r in rows[-top_n_each:]:
            r["source"] = "topic_match"

        # (B) embedding similarity
        rows += s.run(CYPHER_EMBED,
                      vec=parts["paragraph_vec"].tolist(),
                      lim=top_n_each
                     ).data()
        for r in rows[-top_n_each:]:
            r["source"] = "embed"

        # (C) one-hop citations
        rows += s.run(CYPHER_1HOP,
                      terms=terms,
                      lim=top_n_each
                     ).data()
        for r in rows[-top_n_each:]:
            r["source"] = "one_hop"
    
    df = pd.DataFrame(rows).drop_duplicates("pid")
    return df
