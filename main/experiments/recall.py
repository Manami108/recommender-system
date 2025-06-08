# ------------------------- recall.py ------------------------
"""Dual recall: BM25 + vector similarity."""
import re, pandas as pd
from typing import Dict
from .config import driver, embedder, EMBED_TOP_K, KEYWORD_TOP_K, MAX_TOTAL_CANDS

_LUCENE_SPECIAL = r'+-&&||!(){}[]^"~*?:\\/'
_ESC = re.compile(f'([{re.escape(_LUCENE_SPECIAL)}])')

def _quote(t:str)->str: return f'"{_ESC.sub(r"\\\\1",t.strip())}"'

def _bm25_q(intent:Dict)->str:
    kws  = intent.get("keywords", [])
    tech = intent.get("technologies", [])
    dom  = intent.get("research_domain", [])
    parts=[f'{_quote(t)}^3' for t in kws+tech]
    if dom: parts.append("("+" OR ".join(_quote(d) for d in dom)+")")
    return " AND ".join(parts) if parts else ""

CYPHER_BM25 = """
CALL db.index.fulltext.queryNodes('paper_fulltext',$q,{limit:$lim})
YIELD node,score RETURN node.id AS pid,node.title AS title,node.year AS year,score AS bm25
"""
CYPHER_EMBED="""
CALL db.index.vector.queryNodes('paper_vec',$k,$vec)
YIELD node,score RETURN node.id AS pid,node.title AS title,node.year AS year,(1.0-score) AS embed
"""

def recall(intent:Dict) -> pd.DataFrame:
    vec = embedder().encode(" ".join(intent["keywords"]),
                            convert_to_numpy=True,normalize_embeddings=True)
    drv = driver(); rows=[]
    with drv.session() as s:
        q=_bm25_q(intent); lim=KEYWORD_TOP_K
        if q:
            rows+=s.run(CYPHER_BM25,q=q,lim=lim).data()
        rows+=s.run(CYPHER_EMBED,k=EMBED_TOP_K,vec=vec.tolist()).data()
    df=pd.DataFrame(rows)
    if "bm25" not in df: df["bm25"]=0.0
    if "embed" not in df: df["embed"]=0.0
    df["recall"]=df[["bm25","embed"]].max(axis=1)
    return df.drop_duplicates("pid").nlargest(MAX_TOTAL_CANDS,"recall")
