# recall_bm25.py
import pandas as pd
from kg_driver import driver, embed

# ── Cypher templates (role-AND strict + fallbacks) ───────
CYPHER_ROLE_AND = """
MATCH (p:Paper)
WHERE EXISTS{ MATCH (p)-[:HAS_TOPIC|HAS_TOPIC]->(:Topic)
              WHERE toLower(_.name) CONTAINS toLower($method) }
  AND EXISTS{ MATCH (p)-[:HAS_TOPIC|HAS_FOS]->(:Topic)
              WHERE toLower(_.name) CONTAINS toLower($solution) }
  AND EXISTS{ MATCH (p)-[:HAS_TOPIC|HAS_FOS]->(:Topic)
              WHERE ANY(t IN $domains WHERE toLower(_.name) CONTAINS t) }
RETURN p.id AS pid, p.title AS title, p.year AS year, 0 AS hop, 1.0 AS sim
LIMIT $lim
"""

CYPHER_BM25 = """
CALL db.index.fulltext.queryNodes('paper_fulltext', $q, {limit:$lim})
YIELD node, score
RETURN node.id AS pid, node.title AS title, node.year AS year, 0 AS hop, score AS sim
"""

def _role_buckets(ctx):
    method   = ctx["main_topic"][0] if isinstance(ctx["main_topic"],list) else ctx["main_topic"]
    solution = ctx["technologies"][0] if ctx["technologies"] else ""
    domains  = [d.lower() for d in ctx["subtopics"]]+[ctx["research_domain"].lower()]
    return method, solution, domains

def build_bm25_query(ctx):
    m,s,d = _role_buckets(ctx)
    q = f'"{m}"^{3 if m else 1}'
    if s: q += f' AND "{s}"^2'
    if d: q += " AND ("+" OR ".join(d)+")"
    return q

def recall_candidates(ctx, limit=50):
    m,s,d = _role_buckets(ctx)
    q_bm  = build_bm25_query(ctx)
    rows=[]
    with driver.session() as sesh:
        rows += sesh.run(CYPHER_ROLE_AND, method=m, solution=s,
                         domains=d, lim=limit).data()
        if len(rows)<8:          # fallback BM25 if strict few
            rows += sesh.run(CYPHER_BM25, q=q_bm, lim=limit*2).data()
    return pd.DataFrame(rows).drop_duplicates("pid")
