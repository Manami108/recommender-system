# recall_bm25.py
# This method doesnt work because only specific keywords (main and so on) has to match with fos or topic. 
import pandas as pd
from kg_driver import driver, embed
import re

CYPHER_ROLE_AND = """
MATCH (p:Paper)
WHERE
  /* METHOD --------------- */
  EXISTS {
      MATCH (p)-[:HAS_TOPIC|HAS_FOS]->(m)
      WHERE 
        ( (m:Topic AND toLower(m.keywords) CONTAINS toLower($method)) 
          OR 
          (m:FieldOfStudy AND toLower(m.name) CONTAINS toLower($method)) 
        )
  }
  /* SOLUTION ------------- */
  AND
  EXISTS {
      MATCH (p)-[:HAS_TOPIC|HAS_FOS]->(s)
      WHERE 
        ( (s:Topic AND toLower(s.keywords) CONTAINS toLower($solution)) 
          OR 
          (s:FieldOfStudy AND toLower(s.name) CONTAINS toLower($solution)) 
        )
  }
  /* DOMAIN/TASK ---------- */
  AND
  EXISTS {
      MATCH (p)-[:HAS_TOPIC|HAS_FOS]->(d)
      WHERE 
        ANY(dom IN $domains WHERE 
            ( (d:Topic AND toLower(d.keywords) CONTAINS dom) 
              OR 
              (d:FieldOfStudy AND toLower(d.name) CONTAINS dom) 
            )
        )
  }
RETURN p.id  AS pid,
       p.title AS title,
       p.year  AS year,
       0       AS hop,
       1.0     AS sim
LIMIT $lim
"""

CYPHER_BM25 = """
CALL db.index.fulltext.queryNodes('paper_fulltext', $q, {limit:$lim})
YIELD node, score
RETURN node.id    AS pid,
       node.title AS title,
       node.year  AS year,
       0          AS hop,
       score      AS sim
"""

def _role_buckets(ctx):
    method   = ctx["main_topic"][0] if isinstance(ctx["main_topic"], list) else ctx["main_topic"]
    solution = ctx["technologies"][0] if ctx["technologies"] else ""
    # include research_domain plus all subtopics, all lower-cased
    domains  = [d.lower() for d in ctx["subtopics"]] + [ctx["research_domain"].lower()]
    return method, solution, domains

# characters that must be escaped inside a Lucene term
_LUCENE_SPECIAL = r'+-&&||!(){}[]^"~*?:\\/'
_esc = re.compile(f'([{re.escape(_LUCENE_SPECIAL)}])')

def _quote(term: str) -> str:
    term = term.strip()
    term = _esc.sub(r'\\\1', term)
    return f'"{term}"'

def build_bm25_query(ctx):
    m, s, d_list = _role_buckets(ctx)
    parts = []
    if m:
        parts.append(f'{_quote(m)}^3')
    if s:
        parts.append(f'{_quote(s)}^2')
    if d_list:
        dom_phrases = [_quote(t) for t in d_list if t]
        parts.append("(" + " OR ".join(dom_phrases) + ")")
    return " AND ".join(parts) if len(parts) > 1 else (parts[0] if parts else "")

def recall_candidates(ctx, limit=50):
    m, s, d = _role_buckets(ctx)
    q_bm    = build_bm25_query(ctx)
    rows    = []
    with driver.session() as sesh:
        # 1) Strict role-AND
        rows1 = sesh.run(
            CYPHER_ROLE_AND,
            method=m,
            solution=s,
            domains=d,
            lim=limit
        ).data()
        rows += rows1

        # 2) If fewer than 8, do BM25 fallback
        if len(rows1) < 8:
            rows += sesh.run(CYPHER_BM25, q=q_bm, lim=limit*2).data()

    return pd.DataFrame(rows).drop_duplicates("pid")
