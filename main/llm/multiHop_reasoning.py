#!/usr/bin/env python
"""
hybrid_rank25_to10.py
  • text ANN  (paper_vec, SciBERT)
  • graph keyword/FoS  (Cypher)
  • structural ANN  (paper_struct_vec, Node2Vec)
  • soft paper_type boost, duplicate bonus
  • final top-10 with scores and source flags
"""

import os, re, json, ast, argparse, warnings, pandas as pd, torch
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

warnings.filterwarnings("ignore", category=UserWarning)

# ─── Neo4j driver ─────────────────────────────────────────────────────────
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI","bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER","neo4j"), os.getenv("NEO4J_PASS","Manami1008"))
)

# ─── SciBERT embedder (text ANN) ───────────────────────────────────────────
print("▶ Loading SciBERT …")
sci_model = SentenceTransformer("allenai/scibert_scivocab_uncased")
sci_model.eval()
embed = lambda txt: sci_model.encode(txt, convert_to_numpy=True,
                                     normalize_embeddings=True).tolist()

def vector_top25(qvec):
    with driver.session() as s:
        rows = s.run("""
          CALL db.index.vector.queryNodes('paper_vec', 25, $q)
          YIELD node, score RETURN node.id AS id, score ORDER BY score
        """, q=qvec).data()
    return [{"id":r["id"], "sim":1-r["score"], "source_vec":True} for r in rows]


# ─── Node2Vec structural ANN (new) ─────────────────────────────────────────
import math

def struct_top15(seed_ids, k=15):
    """Try each candidate id until we find a non-zero n2v vector."""
    with driver.session() as s:
        for pid in seed_ids:
            rec = s.run("MATCH (p:Paper {id:$pid}) RETURN p.n2v AS v", pid=pid).single()
            vec = rec and rec["v"]
            if not vec:                       # null property
                continue
            if isinstance(vec, str):         # if imported as string "0;0;…"
                vec = [float(x) for x in vec.split(";")]
            if math.isfinite(sum(vec)) and any(abs(x) > 1e-6 for x in vec):
                rows = s.run("""
                  CALL db.index.vector.queryNodes('paper_struct_vec', $k, $v)
                  YIELD node, score
                  RETURN node.id AS id, score ORDER BY score
                """, k=k, v=vec).data()
                return [{"id":r["id"], "sim_struct":1-r["score"], "source_struct":True}
                        for r in rows]
    return []   # all candidates had zero vec

# ─── Fuzzy concept → Topic / FoS mapping via vector ANN ────────────────
SIM_THRESHOLD = 0.50   # minimum cosine-similarity to accept a match

def nearest_topic_fos(term: str):
    """Return ('topic', idx) or ('fos', idx) if similarity ≥ threshold; else (None,None)."""
    vec = embed(term)                                   # 768-d paragraph encoder
    with driver.session() as s:
        t = s.run("""
            CALL db.index.vector.queryNodes('topic_vec', 1, $v)
            YIELD node, score RETURN node.idx AS idx, score
        """, v=vec).single()
        f = s.run("""
            CALL db.index.vector.queryNodes('fos_vec', 1, $v)
            YIELD node, score RETURN node.idx AS idx, score
        """, v=vec).single()

    cand = []
    if t: cand.append(("topic", 1 - t["score"], int(t["idx"])))
    if f: cand.append(("fos",   1 - f["score"], int(f["idx"])))
    if not cand:         return None, None
    kind, sim, idx = max(cand, key=lambda x: x[1])
    if sim < SIM_THRESHOLD:  return None, None
    return kind, idx

# ─── LLaMA concept extractor (unchanged) ───────────────────────────────────
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
print("▶ Loading LLaMA-8B …")
llm = pipeline("text-generation",
    model=AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.float16),
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID)
)
PROMPT = """You are an academic assistant… JSON: {{"concepts":["x"],"paper_type":"survey"}} <TEXT>{paragraph}</TEXT>"""

def llm_extract(paragraph:str):
    out = llm(PROMPT.format(paragraph=paragraph.strip()),
              max_new_tokens=256,temperature=0.0,do_sample=False)[0]["generated_text"]
    for chunk in re.findall(r"\{[^{}]+\}", out, re.S):
        for loader in (json.loads, ast.literal_eval):
            try:
                d = loader(chunk)
                if "concepts" in d and "paper_type" in d: return d
            except: pass
    raise ValueError("LLM output unparsable:\n"+out)

# ─── Graph keyword/FoS branch (same as before) ─────────────────────────────
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

def build_id_lists(concepts):
    topic_ids, fos_ids = set(), set()
    for c in concepts:
        if not isinstance(c, str) or len(c.split()) < 2:
            continue
        kind, idx = nearest_topic_fos(c)
        if kind == "topic": topic_ids.add(idx)
        elif kind == "fos": fos_ids.add(idx)
    return list(topic_ids), list(fos_ids)

CYPHER_HITS = """
MATCH (p:Paper)
OPTIONAL MATCH (p)-[:HAS_TOPIC]->(t:Topic)
  WHERE t.idx IN $topicIds
OPTIONAL MATCH (p)-[:HAS_FOS]->(f:FieldOfStudy)
  WHERE f.idx IN $fosIds
WITH p, count(DISTINCT t) + count(DISTINCT f) AS total_hits
WHERE total_hits > 0
RETURN p.id AS id, p.title AS title, p.year AS year, total_hits,
       CASE $ptype
         WHEN 'survey'        THEN (toLower(p.title) CONTAINS 'survey'   OR toLower(p.abstract) CONTAINS 'survey')
         WHEN 'benchmark'     THEN (toLower(p.title) CONTAINS 'benchmark'OR toLower(p.abstract) CONTAINS 'benchmark')
         WHEN 'method paper'  THEN (toLower(p.abstract) CONTAINS 'we propose' OR toLower(p.abstract) CONTAINS 'novel method')
         WHEN 'application'   THEN (toLower(p.abstract) CONTAINS 'we apply'   OR toLower(p.abstract) CONTAINS 'application')
         WHEN 'theoretical'   THEN (toLower(p.abstract) CONTAINS 'we prove'   OR toLower(p.abstract) CONTAINS 'theorem')
         ELSE false END AS type_match
ORDER BY total_hits DESC, p.year DESC
LIMIT 50;
"""

def graph_top50_fuzzy(concepts, ptype):
    topicIds, fosIds = build_id_lists(concepts)
    if not topicIds and not fosIds:
        return []                       # rely on ANN branches only
    with driver.session() as s:
        rows = s.run(CYPHER_HITS,
                     topicIds=topicIds, fosIds=fosIds, ptype=ptype).data()
    out=[]
    for r in rows:
        out.append({"id":r["id"],"title":r["title"],"year":r["year"],
                    "hits":r["total_hits"],
                    "type_match":bool(r["type_match"]),
                    "source_graph":True})
    return out

# ─── Merge + scoring (add struct) ─────────────────────────────────────────
def score_and_rank(text_rows, graph_rows, struct_rows, ptype, top=10):
    pool={}
    for r in text_rows+graph_rows+struct_rows:
        e=pool.setdefault(r["id"],{"id":r["id"],
                                   "source_vec":False,"source_graph":False,"source_struct":False,
                                   "sim":0,"sim_struct":0,"type_match":False, "hits": 0})
        if r.get("source_vec"):   e["source_vec"]=True;   e["sim"]=max(e["sim"], r["sim"])
        if r.get("source_struct"):e["source_struct"]=True;e["sim_struct"]=max(e["sim_struct"], r["sim_struct"])
        if r.get("source_graph"):
            e["source_graph"]=True
            e.update({k:r[k] for k in ("title","year") if k in r})
            e["type_match"]|=r.get("type_match",False)
        if r.get("hits") is not None:
            e["hits"] = max(e["hits"], r["hits"])
            

    ranked=[]
    for d in pool.values():
        sc = 0.0
        sc += d["sim"]                          # text similarity
        sc += 0.10*d["sim_struct"]
        sc += 0.10 * d.get("hits", 0)              # multi-topic coverage# structural similarity (scaled)
        if d["source_graph"]: sc += 0.20
        if d["source_vec"] and d["source_graph"]: sc += 1.0
        if ptype!="any" and d["type_match"]: sc += 0.15
        d["score"]=sc; ranked.append(d)

    ranked.sort(key=lambda x:x["score"], reverse=True)

    # hydrate metadata for missing titles
    miss=[r["id"] for r in ranked[:25] if "title" not in r]
    if miss:
        with driver.session() as s:
            meta=s.run("MATCH (p:Paper) WHERE p.id IN $ids RETURN p.id AS id,p.title AS title,p.year AS year", ids=miss).data()
        m={d["id"]:d for d in meta}
        for r in ranked:
            if "title" not in r: r.update(m.get(r["id"], {"title":"(missing)","year":""}))

    df=pd.DataFrame(ranked[:top])
    for col in ("source_vec","source_graph","source_struct"):  # guarantee cols
        if col not in df.columns: df[col]=False
    return df

# ─── CLI -------------------------------------------------------------------
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("-p","--paragraph",required=True)
    args=p.parse_args()

    # LLM first
    info=llm_extract(args.paragraph)
    print("\nLLM extraction:", info)
    graph_rows = graph_top50_fuzzy(info["concepts"], info["paper_type"])


    # text ANN
    vec_rows = vector_top25(embed(args.paragraph))

    # choose first text-paper as anchor for structural ANN
    seed_ids = [r["id"] for r in vec_rows]        # ordered by text similarity
    struct_rows = struct_top15(seed_ids)          # may return []


    top10 = score_and_rank(vec_rows, graph_rows, struct_rows, info["paper_type"])

    if top10.empty:
        print("\n⚠ No papers found.")
    else:
        print("\nTop-10 ranked papers:\n")
        print(top10[["id","title","year","score","hits",
                    "source_vec","source_struct","source_graph"]]
              .to_markdown(index=False))


# run command 
# python a.py -p "Therefore, knowledge graphs have seized great opportunities by improving the quality of AI systems and being applied to various areas. However, the research on knowledge graphs still faces significant technical challenges. For example, there are major limitations in the current technologies for acquiring knowledge from multiple sources and integrating them into a typical knowledge graph. Thus, knowledge graphs provide great opportunities in modern society. However, there are technical challenges in their development. Consequently, it is necessary to analyze the knowledge graphs with respect to their opportunities and challenges to develop a better understanding of the knowledge graphs."

# python a.py -p "An alternative to directed graphical models with latent variables are undirected graphical models with latent variables, such as restricted Boltzmann machines (RBMs), deep Boltzmann machines (DBMs) and their numerous variants. The interactions within such models are represented as the product of unnormalized potential functions, normalized by a global summation/integration over all states of the random variables. This quantity (the partition function) and its gradient are intractable for all but the most trivial instances, although they can be estimated by Markov chain Monte Carlo (MCMC) methods. Mixing poses a significant problem for learning algorithms that rely on MCMC."
# python a.py -p "In image recognition, VLAD is a representation that encodes by the residual vectors with respect to a dictionary, and Fisher Vector [30] can be formulated as a probabilistic version [18] of VLAD. Both of them are powerful shallow representations for image retrieval and classification [4, 48]. For vector quantization, encoding residual vectors [17] is shown to be more effective than encoding original vectors. In low-level vision and computer graphics, for solving Partial Differential Equations (PDEs), the widely used Multigrid method [3] reformulates the system as subproblems at multiple scales, where each subproblem is responsible for the residual solution between a coarser and a finer scale. An alternative to Multigrid is hierarchical basis preconditioning [45, 46], which relies on variables that represent residual vectors between two scales. It has been shown [3, 45, 46] that these solvers converge much faster than standard solvers that are unaware of the residual nature of the solutions. These methods suggest that a good reformulation or preconditioning can simplify the optimization."