#!/usr/bin/env python
"""
hybrid_rank25_to10.py
Fetch 25 papers from vector search + 25 from graph search,
merge, score, and output the top 10 results.
"""

import os, re, json, ast, argparse, warnings
import pandas as pd, torch
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

warnings.filterwarnings("ignore", category=UserWarning)

# ─── Neo4j driver ───────────────────────────────────────────────────────────
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI",  "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER","neo4j"), os.getenv("NEO4J_PASS","Manami1008"))
)

# ─── SciBERT embedder ───────────────────────────────────────────────────────
print("▶ Loading SciBERT …")
sci_model = SentenceTransformer("allenai/scibert_scivocab_uncased")
sci_model.eval()

def embed(text:str):
    return sci_model.encode(text, convert_to_numpy=True,
                            normalize_embeddings=True).tolist()

def vector_top25(qvec):
    with driver.session() as s:
        rows = s.run("""
            CALL db.index.vector.queryNodes('paper_vec', 25, $q)
            YIELD node, score
            RETURN node.id AS id, score
            ORDER BY score ASC    // cosine distance: smaller is nearer
        """, q=qvec).data()
    return [{"id":r["id"],
             "sim": 1.0 - r["score"],   # similarity ∈ [−1,1]
             "source_vec":True} for r in rows]

# ─── LLaMA extractor ────────────────────────────────────────────────────────
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
print("▶ Loading LLaMA-8B …")
llm = pipeline("text-generation",
    model=AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.float16),
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID)
)

PROMPT = """
You are an academic assistant helping a researcher find papers.
Extract JSON: {{"concepts":["x","y"],"paper_type":"survey"}}
<TEXT>{paragraph}</TEXT>
"""

def llm_extract(paragraph:str):
    out = llm(PROMPT.format(paragraph=paragraph.strip()),
              max_new_tokens=256,temperature=0.0,do_sample=False)[0]["generated_text"]
    for chunk in re.findall(r"\{[^{}]+\}", out, re.S):
        for loader in (json.loads, ast.literal_eval):
            try:
                d = loader(chunk);  # may raise
                if "concepts" in d and "paper_type" in d:
                    return d
            except Exception:
                continue
    raise ValueError("LLM output unparsable:\n"+out)

# ─── Cypher branch ──────────────────────────────────────────────────────────
TYPE_FILTER = {
  "survey":"AND (toLower(p.title) CONTAINS 'survey' OR toLower(p.abstract) CONTAINS 'survey')",
  "benchmark":"AND (toLower(p.title) CONTAINS 'benchmark' OR toLower(p.abstract) CONTAINS 'benchmark')",
  "method paper":"AND (toLower(p.abstract) CONTAINS 'we propose' OR toLower(p.abstract) CONTAINS 'novel method')",
  "application":"AND (toLower(p.abstract) CONTAINS 'we apply' OR toLower(p.abstract) CONTAINS 'application')",
  "theoretical":"AND (toLower(p.abstract) CONTAINS 'we prove' OR toLower(p.abstract) CONTAINS 'theorem')",
  "any":""
}

CYPHER = """
MATCH (p:Paper)
WHERE EXISTS {{
  MATCH (p)-[:HAS_TOPIC]->(t:Topic)
  WHERE ANY(term IN $terms WHERE toLower(t.keywords) CONTAINS term)
}}
OR EXISTS {{
  MATCH (p)-[:HAS_FOS]->(f:FieldOfStudy)
  WHERE ANY(term IN $terms WHERE toLower(f.name) = term)
}}
RETURN p.id   AS id,
       p.title AS title,
       p.year  AS year,
       // soft match for paper_type
       CASE $ptype
         WHEN 'survey'        THEN (toLower(p.title)  CONTAINS 'survey'   OR toLower(p.abstract) CONTAINS 'survey')
         WHEN 'benchmark'     THEN (toLower(p.title)  CONTAINS 'benchmark'OR toLower(p.abstract) CONTAINS 'benchmark')
         WHEN 'method paper'  THEN (toLower(p.abstract) CONTAINS 'we propose' OR toLower(p.abstract) CONTAINS 'novel method')
         WHEN 'application'   THEN (toLower(p.abstract) CONTAINS 'we apply'   OR toLower(p.abstract) CONTAINS 'application')
         WHEN 'theoretical'   THEN (toLower(p.abstract) CONTAINS 'we prove'   OR toLower(p.abstract) CONTAINS 'theorem')
         ELSE false
       END AS type_match
ORDER BY p.year DESC
LIMIT 50;
"""


def graph_top50(concepts, ptype):
    terms = [t.lower() for t in concepts if isinstance(t,str) and len(t.split())>=2]
    if not terms: return []
    with driver.session() as s:
        rows = s.run(CYPHER, terms=terms, ptype=ptype).data()
    out = []
    for r in rows:
        out.append({"id": r["id"],
                    "title": r["title"],
                    "year": r["year"],
                    "type_match": bool(r["type_match"]),
                    "source_graph": True})
    return out

# ─── Merge, score, pick top-10 ──────────────────────────────────────────────
def hydrate_meta(rows_missing):
    ids = [r["id"] for r in rows_missing]
    with driver.session() as s:
        meta = s.run("MATCH (p:Paper) WHERE p.id IN $ids "
                     "RETURN p.id AS id, p.title AS title, p.year AS year",
                     ids=ids).data()
    meta_d = {m["id"]:m for m in meta}
    for r in rows_missing:
        r.update(meta_d.get(r["id"], {"title":"(missing)","year":""}))

# ─── Merge, score, pick top-10 (fixed) ───────────────────────────────
def score_and_rank(vec_rows, graph_rows, ptype, top=10):
    pool = {}
    for r in vec_rows + graph_rows:
        entry = pool.setdefault(r["id"], {
            "id": r["id"], "source_vec": False, "source_graph": False,
            "sim": 0.0, "type_match": False})
        if r.get("source_vec"):
            entry["source_vec"] = True
            entry["sim"] = max(entry["sim"], r.get("sim", 0))
        if r.get("source_graph"):
            entry["source_graph"] = True
            entry.update({k: r[k] for k in ("title", "year") if k in r})
            entry["type_match"] |= r.get("type_match", False)

    ranked = []
    for d in pool.values():
        sc = 0.0
        if d["source_vec"]:   sc += d["sim"]
        if d["source_graph"]: sc += 0.20
        if d["source_vec"] and d["source_graph"]: sc += 1.0
        if ptype != "any" and d["type_match"]:   sc += 0.15
        d["score"] = sc
        ranked.append(d)

    ranked.sort(key=lambda x: x["score"], reverse=True)

    # hydrate metadata for vector-only rows (max first 25)
    missing = [r["id"] for r in ranked[:25] if "title" not in r]
    if missing:
        with driver.session() as s:
            meta = s.run("""
                MATCH (p:Paper) WHERE p.id IN $ids
                RETURN p.id AS id, p.title AS title, p.year AS year
            """, ids=missing).data()
        meta_map = {m["id"]: m for m in meta}
        for r in ranked:
            if "title" not in r:
                r.update(meta_map.get(r["id"], {"title": "(missing)", "year": ""}))

    df = pd.DataFrame(ranked[:top])

    # ensure indicator columns exist
    for col in ("source_vec", "source_graph"):
        if col not in df.columns:
            df[col] = False
    return df



# ─── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p","--paragraph",required=True)
    args = ap.parse_args()

    # symbolic first (we may want ptype before scoring)
    info = llm_extract(args.paragraph)
    print("\nLLM extraction:", info)
    graph_rows = graph_top50(info["concepts"], info["paper_type"])

    # semantic
    vec25 = vector_top25(embed(args.paragraph))

    # rank
    top10 = score_and_rank(vec25, graph_rows, info["paper_type"])

    if top10.empty:
        print("\n⚠  No papers found.")
    else:
        print("\nTop 10 ranked papers:\n")
        print(top10[["id","title","year","score",
                     "source_vec","source_graph"]].to_markdown(index=False))


# run command 
# python hybrid_query.py -p "Therefore, knowledge graphs have seized great opportunities by improving the quality of AI systems and being applied to various areas. However, the research on knowledge graphs still faces significant technical challenges. For example, there are major limitations in the current technologies for acquiring knowledge from multiple sources and integrating them into a typical knowledge graph. Thus, knowledge graphs provide great opportunities in modern society. However, there are technical challenges in their development. Consequently, it is necessary to analyze the knowledge graphs with respect to their opportunities and challenges to develop a better understanding of the knowledge graphs."