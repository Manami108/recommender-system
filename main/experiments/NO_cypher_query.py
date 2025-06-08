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

# ----------  BM25 helper -------------------------------------------------
def build_bm25_query(ctx: dict) -> str:
    """
    Convert method/domain/solution roles into a Lucene query string.
    """
    method   = ctx["main_topic"][0] if isinstance(ctx["main_topic"], list) else ctx["main_topic"]
    solution = ctx["technologies"][0] if ctx["technologies"] else ""
    domains  = ctx["subtopics"] + [ctx["research_domain"]]

    def q(x):            # quote multi-word phrases
        return f'"{x}"' if " " in x else x

    parts = []
    if method:
        parts.append(f"{q(method)}^3")           # strongest boost
    if solution:
        parts.append(f"{q(solution)}^2")
    dom_tokens = [q(d) for d in domains if d]
    if dom_tokens:
        parts.append("(" + " OR ".join(dom_tokens) + ")")

    return " AND ".join(parts) if len(parts) > 1 else parts[0]

CYPHER_ROLE_AND = """
MATCH (p:Paper)
WHERE
  // METHOD  – main_topic  (e.g. “large language model”)
  EXISTS {
      MATCH (p)-[:HAS_TOPIC|HAS_TOPIC]->(m)
      WHERE toLower(m.name) CONTAINS toLower($method)
  }
  AND
  // SOLUTION – first technology  (e.g. “knowledge graph”)
  EXISTS {
      MATCH (p)-[:HAS_TOPIC|HAS_FOS]->(s)
      WHERE toLower(s.name) CONTAINS toLower($solution)
  }
  AND
  // DOMAIN/TASK – any subtopic or research_domain  (e.g. “recommender”, “tourism”)
  EXISTS {
      MATCH (p)-[:HAS_TOPIC|HAS_FOS]->(d)
      WHERE ANY(dom IN $domains WHERE toLower(d.name) CONTAINS dom)
  }
RETURN p.id  AS pid,
       p.title AS title,
       p.year  AS year,
       0       AS hop,
       1.0     AS sim          // highest confidence
LIMIT $lim
"""
def _role_terms(ctx):
    method   = ctx["main_topic"][0] if isinstance(ctx["main_topic"], list) else ctx["main_topic"]
    solution = ctx["technologies"][0] if ctx["technologies"] else ""
    # keep only ≥2-token domain phrases for precision
    domains  = [d.lower() for d in ctx["subtopics"] + [ctx["research_domain"]] if len(d.split()) >= 2]
    return method, solution, domains


# ----------  BM25 Cypher template ---------------------------------------
CYPHER_BM25 = """
CALL db.index.fulltext.queryNodes(
        'paper_fulltext',
        $lucene,
        {limit: $lim}            // map, not scalar
)
YIELD node, score
RETURN node.id    AS pid,
       node.title AS title,
       node.year  AS year,
       0          AS hop,
       score      AS sim
"""

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
MATCH (p:Paper)-[:HAS_FOS|HAS_TOPIC]->(x)
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
MATCH (seed:Paper)-[:HAS_FOS|HAS_TOPIC]->(x)
WHERE ANY(t IN $terms WHERE toLower(x.name) CONTAINS t)
MATCH (seed)-[:CITES]->(p:Paper)
RETURN p.id  AS pid, p.title AS title, p.year AS year,
       2 AS hop, 0.5 AS sim
LIMIT $lim
"""

def retrieve_candidates(ctx: dict, top_n_each: int = 40) -> pd.DataFrame:
    parts    = build_term_lists(ctx)
    method, solution, domains = _role_terms(ctx)
    bm25_query = build_bm25_query(ctx)
    rows = []

    with driver.session() as s:
        # ── Tier-1  strict role AND  ───────────────────────────────────
        rows1 = s.run(
            CYPHER_ROLE_AND,
            method=method,
            solution=solution,
            domains=domains,
            lim=top_n_each
        ).data()
        for r in rows1:
            r["source"] = "role_and"
        rows += rows1

        # if strict match produced ≥8 papers we can stop early
        if len(rows1) < 8:
            # ── fallback (your original three branches) ───────────────
            terms = parts["keywords"] or [method.lower()]
            rows += s.run(CYPHER_FOS_TOPIC, terms=terms, lim=top_n_each).data()
            for r in rows[-top_n_each:]:
                r["source"] = "topic_match"

            rows += s.run(CYPHER_EMBED,
                          vec=parts["paragraph_vec"].tolist(),
                          lim=top_n_each).data()
            for r in rows[-top_n_each:]:
                r["source"] = "embed"

            rows += s.run(CYPHER_1HOP, terms=terms, lim=top_n_each).data()
            for r in rows[-top_n_each:]:
                r["source"] = "one_hop"

            rows += s.run(CYPHER_BM25,
                          lucene=bm25_query,
                          lim=top_n_each).data()
            for r in rows[-top_n_each:]:
                r["source"] = "bm25"

    # ── combine & dedupe ───────────────────────────────────────────────
    df = pd.DataFrame(rows).drop_duplicates("pid")

    # optional: simple final score (role-AND first)
    df["score"] = (
          (df["source"] == "role_and") * 1.0
        + (df["source"] == "topic_match") * 0.8
        + (df["source"] == "bm25") * (df["sim"] / df["sim"].max())
        + (df["source"] == "embed") * 0.6 * df["sim"]
        + (df["source"] == "one_hop") * 0.3
    )
    return df.sort_values("score", ascending=False).head( top_n_each * 2 )
