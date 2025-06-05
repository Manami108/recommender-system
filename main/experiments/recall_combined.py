# recall_combined.py

import pandas as pd
import re
from kg_driver import driver, embed
import numpy as np

# ─── Cypher template for Method A: single-OR BM25 on title/abstract ─────
CYPHER_BM25_OR = """
CALL db.index.fulltext.queryNodes(
    'paper_fulltext',
    $q,
    { limit: $lim }
)
YIELD node, score
RETURN
    node.id    AS pid,
    node.title AS title,
    node.year  AS year,
    0          AS hop,
    score      AS sim
LIMIT $lim
"""

# ─── Cypher template for Method B: embedding similarity ───────────────────
CYPHER_EMBED = """
CALL db.index.vector.queryNodes('paper_vec', $lim, $vec)
YIELD node, score
RETURN
    node.id    AS pid,
    node.title AS title,
    node.year  AS year,
    0          AS hop,
    1.0 - score AS sim   // higher sim = closer in embedding space
LIMIT $lim
"""

# Characters that must be escaped inside a Lucene term
_LUCENE_SPECIAL = r'+-&&||!(){}[]^"~*?:\\/'
_esc            = re.compile(f'([{re.escape(_LUCENE_SPECIAL)}])')

def _quote(term: str) -> str:
    """
    Surround the term with quotes and escape any Lucene‐special characters inside.
    Example:  / → \/ ;  "hello" → \"hello\" 
    """
    t = term.strip()
    t = _esc.sub(r'\\\1', t)
    return f'"{t}"'

def build_or_bm25_query(ctx: dict) -> str:
    """
    Construct a single Lucene query string that OR's together:
      • main_topic^3
      • each technology^2
      • each subtopic^1
      • research_domain^1

    If there is only one phrase, it returns that phrase (with optional boost).
    """
    parts = []
    # 1) main_topic boosted ×3
    main_topic = ctx["main_topic"][0] if isinstance(ctx["main_topic"], list) else ctx["main_topic"]
    if main_topic:
        parts.append(f"{_quote(main_topic)}^3")

    # 2) each technology boosted ×2
    for tech in ctx["technologies"]:
        parts.append(f"{_quote(tech)}^2")

    # 3) each subtopic (no boost)
    for sub in ctx["subtopics"]:
        parts.append(_quote(sub))

    # 4) research_domain (no boost)
    if ctx["research_domain"]:
        parts.append(_quote(ctx["research_domain"]))

    # Join them all by OR.  If there's exactly one part, return that; otherwise OR‐join.
    if len(parts) == 0:
        return ""    # no terms at all
    elif len(parts) == 1:
        return parts[0]
    else:
        return " OR ".join(parts)

def recall_combined(ctx: dict, limit: int = 50) -> pd.DataFrame:
    """
    1) Run BM25‐OR (Method A) to get up to `limit` candidates.
    2) Run embedding‐NN (Method B) to get up to `limit` candidates.
    3) Union them (drop duplicates by PID) and return the merged DataFrame.
    """
    # 1) Build the OR‐BM25 query
    q_bm = build_or_bm25_query(ctx)

    # 2) Build the paragraph embedding for Method B
    #    We combine problem_statement + subtopics + technologies into one string
    paragraph_text = " ".join([
        ctx["problem_statement"],
        " ".join(ctx["subtopics"]),
        " ".join(ctx["technologies"])
    ]).strip()
    p_vec = embed(paragraph_text)  # returns a numpy array (normalized)

    rows = []
    with driver.session() as sess:
        # ─── Method A: OR‐BM25 on title/abstract ─────────────────────────
        if q_bm:
            bm25_rows = sess.run(
                CYPHER_BM25_OR,
                q=q_bm,
                lim=limit
            ).data()
        else:
            bm25_rows = []

        for r in bm25_rows:
            # tag the source so we know it came from the OR‐BM25 branch
            r["source"] = "bm25_or"
        rows.extend(bm25_rows)

        # ─── Method B: embedding‐NN ──────────────────────────────────────
        embed_rows = sess.run(
            CYPHER_EMBED,
            vec=p_vec.tolist(),
            lim=limit
        ).data()

        for r in embed_rows:
            # tag the source so we know it came from the embedding branch
            r["source"] = "embed_nn"
        rows.extend(embed_rows)

    # 3) Combine & deduplicate by pid
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Drop duplicates, keeping the row with the higher “sim” if the same pid appears twice
    df = df.sort_values("sim", ascending=False).drop_duplicates("pid", keep="first")
    return df.reset_index(drop=True)
