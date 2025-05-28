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

# --- replace the whole function ---------------------------------
def nearest_topic_fos(vec):
    """Return ('topic'|'fos', idx) or (None,None)"""
    with driver.session() as s:
        t = s.run(
            "CALL db.index.vector.queryNodes('topic_vec', 1, $v) "
            "YIELD node, score RETURN node.idx AS idx, score",
            v=vec).single()

        f = s.run(
            "CALL db.index.vector.queryNodes('fos_vec', 1, $v) "
            "YIELD node, score RETURN node.idx AS idx, score",
            v=vec).single()

    cand = []
    if t: cand.append(("topic", 1 - t["score"], int(t["idx"])))
    if f: cand.append(("fos",   1 - f["score"], int(f["idx"])))
    if not cand:
        return None, None

    kind, sim, idx = max(cand, key=lambda x: x[1])   # higher sim first
    return (kind, idx) if sim >= SIM_THRESHOLD else (None, None)




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

def build_id_lists(concept_vecs):
    topic_ids, fos_ids = set(), set()
    for vec in concept_vecs:
        kind, idx = nearest_topic_fos(vec)
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

def graph_top50_fuzzy(concept_vecs, ptype):
    topicIds, fosIds = build_id_lists(concept_vecs)
    if not topicIds and not fosIds:
        return []
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
# ─── EXTRA: utility tokenizer for whitelist ──────────────────────────────
from transformers import AutoTokenizer
bert_tok = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

def make_whitelist_tokens(texts, extra_terms):
    vocab = set()
    for txt in texts + extra_terms:
        for tok in bert_tok.tokenize(txt):
            vocab.add(tok)
    # also add common punctuation + stop words
    vocab.update([",", ".", "(", ")", "the", "of", "and", "to", "is", "in", "for"])
    ids = [bert_tok.convert_tokens_to_ids(t) for t in vocab if t in bert_tok.vocab]
    return set(ids)

# ─── SHORT PATH → template sentence (≤3 hops) ────────────────────────────
def shortest_path_sentence(pid, topicIds, fosIds):
    query = """
    MATCH (p:Paper {id:$pid}),
          (x)
    WHERE (x:Topic AND x.idx IN $topicIds) OR (x:FieldOfStudy AND x.idx IN $fosIds)
    WITH p,x LIMIT 1
    MATCH pth = shortestPath( (p)-[*..3]-(x) )
    RETURN [n IN nodes(pth) | labels(n)[0]] AS labs,
           [n IN nodes(pth) | coalesce(n.title, n.keywords, n.name, n.idx)] AS names
    """
    rec = driver.session().run(query, pid=pid, topicIds=topicIds, fosIds=fosIds).single()
    if not rec: return ""
    labs, names = rec["labs"], rec["names"]
    # simple verbalization
    hops = " → ".join(f"{names[i]} ({labs[i]})" for i in range(len(names)))
    return f"Path: {hops}"

# ─── LLaMA whitelist-constrained summary  (top-3) ─────────────────────────
def safe_summary(title, abstract, paragraph, whitelist_ids, timeout=2.0):
    prompt = f"""Context:
TITLE: {title}
ABSTRACT: {abstract}

Paragraph: {paragraph}

Write two sentences explaining why this paper is relevant. Use ONLY words from the context."""
    try:
        out = llm(prompt,
                  max_new_tokens=60,
                  temperature=0.0,
                  do_sample=False,
                  timeout=timeout)[0]["generated_text"]

        # guard: every sub-token must be in whitelist
        for tok in out.split():
            if bert_tok.convert_tokens_to_ids(tok) not in whitelist_ids:
                raise ValueError("OOV token")
        return out.strip()
    except Exception:
        # fallback → first two sentences of abstract
        sent = abstract.split(". ")[:2]
        return ". ".join(sent).strip()


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

    for col in ("summary", "explain_path"):
        if col not in df.columns:
            df[col] = ""
    return df


# ─── CLI -------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--paragraph", required=True,
                    help="Input paragraph.")
    args = ap.parse_args()

    # 1) LLaMA extraction
    info = llm_extract(args.paragraph)
    print("\nLLM extraction:", info)

    # 2) ONE SciBERT call (paragraph + concepts)  ─────────────
    texts = [args.paragraph] + info["concepts"]          # length = 1 + N_concepts
    vecs  = sci_model.encode(texts, convert_to_numpy=True,
                             normalize_embeddings=True)  # batch embed
    qvec, concept_vecs = vecs[0], vecs[1:]               # split

    # 3) Compute text ANN (you were missing this!)
    vec_rows = vector_top25(qvec)


    # 6) Structural ANN branch (seed = text-similar papers)
    seed_ids    = [r["id"] for r in vec_rows]
    struct_rows = struct_top15(seed_ids)                 # may be []

    # 3) Build fuzzy Topic / FoS ID lists from concept vectors
    topicIds, fosIds = build_id_lists(concept_vecs)
    
    # 4) Graph branch (Cypher hits)
    graph_rows = graph_top50_fuzzy(concept_vecs, info["paper_type"])
    
    # 7) Fuse + rank
    top_df = score_and_rank(vec_rows, graph_rows, struct_rows,
                            info["paper_type"])



    # 8) Explanations  (path + summary)
    for idx, row in top_df.iterrows():
        top_df.at[idx, "explain_path"] = shortest_path_sentence(
            row["id"], topicIds, fosIds)

    # ---- batch summaries for top-3 ----------------------------------
    top3 = top_df.head(3).index
    prompts, whitelists = [], []
    for idx in top3:
        rec = driver.session().run(
            "MATCH (p:Paper {id:$pid}) RETURN p.title AS t, p.abstract AS a",
            pid=top_df.at[idx, "id"]).single()
        wl = make_whitelist_tokens([rec["t"], rec["a"]], info["concepts"])
        whitelists.append(wl)
        prompts.append(
            f"Context:\nTITLE: {rec['t']}\nABSTRACT: {rec['a']}\n\n"
            f"Paragraph: {args.paragraph}\n\n"
            "Write two sentences explaining why this paper is relevant. "
            "Use ONLY words from the context."
        )

    if prompts:
        outs = llm(prompts, max_new_tokens=60, temperature=0.0, do_sample=False)
        for (idx, wl, gen) in zip(top3, whitelists, outs):
            txt = gen[0]["generated_text"].strip() 
            # whitelist guard
            ok = all(bert_tok.convert_tokens_to_ids(tok) in wl for tok in txt.split())
            if ok:
                top_df.at[idx, "summary"] = txt
            else:
                # fallback to abstract snippet
                abs_snip = ". ".join(prompts[top3.get_loc(idx)]
                                     .split("ABSTRACT: ")[1]
                                     .split("\n")[0]
                                     .split(". ")[:2]).strip()
                top_df.at[idx, "summary"] = abs_snip or "(abstract unavailable)"

    # summaries for rows 4-10
    for idx in top_df.index.difference(top3):
        rec = driver.session().run(
            "MATCH (p:Paper {id:$pid}) RETURN p.abstract AS a",
            pid=top_df.at[idx, "id"]).single()
        snip = ". ".join(rec["a"].split(". ")[:2]).strip()
        top_df.at[idx, "summary"] = snip or "(abstract unavailable)"

    # 9) Display
    print("\nTop-10 ranked papers:\n")
    print(
        top_df[["id", "title", "year", "score",
                "source_vec", "source_struct", "source_graph",
                "hits", "summary", "explain_path"]]
        .to_markdown(index=False)
    )

# run command 
# python time_minimize.py -p "Therefore, knowledge graphs have seized great opportunities by improving the quality of AI systems and being applied to various areas. However, the research on knowledge graphs still faces significant technical challenges. For example, there are major limitations in the current technologies for acquiring knowledge from multiple sources and integrating them into a typical knowledge graph. Thus, knowledge graphs provide great opportunities in modern society. However, there are technical challenges in their development. Consequently, it is necessary to analyze the knowledge graphs with respect to their opportunities and challenges to develop a better understanding of the knowledge graphs."

# python time_minimize.py -p "An alternative to directed graphical models with latent variables are undirected graphical models with latent variables, such as restricted Boltzmann machines (RBMs), deep Boltzmann machines (DBMs) and their numerous variants. The interactions within such models are represented as the product of unnormalized potential functions, normalized by a global summation/integration over all states of the random variables. This quantity (the partition function) and its gradient are intractable for all but the most trivial instances, although they can be estimated by Markov chain Monte Carlo (MCMC) methods. Mixing poses a significant problem for learning algorithms that rely on MCMC."

# python time_minimize.py -p "In image recognition, VLAD is a representation that encodes by the residual vectors with respect to a dictionary, and Fisher Vector [30] can be formulated as a probabilistic version [18] of VLAD. Both of them are powerful shallow representations for image retrieval and classification [4, 48]. For vector quantization, encoding residual vectors [17] is shown to be more effective than encoding original vectors. In low-level vision and computer graphics, for solving Partial Differential Equations (PDEs), the widely used Multigrid method [3] reformulates the system as subproblems at multiple scales, where each subproblem is responsible for the residual solution between a coarser and a finer scale. An alternative to Multigrid is hierarchical basis preconditioning [45, 46], which relies on variables that represent residual vectors between two scales. It has been shown [3, 45, 46] that these solvers converge much faster than standard solvers that are unaware of the residual nature of the solutions. These methods suggest that a good reformulation or preconditioning can simplify the optimization."