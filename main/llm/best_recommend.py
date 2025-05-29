#!/usr/bin/env python
# Hybrid Paper Recommender (rank-based fusion)

import os, re, json, ast, argparse, warnings, math, pandas as pd, torch
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────── Parameters ────────────────────────────────────────
TOP_N = 50          # how many candidates to take from EACH branch
RRF_K = 60          # constant in Reciprocal-Rank Fusion  (larger ⇒ smoother)

# ─────────── Connections ───────────────────────────────────────
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI",  "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"),
          os.getenv("NEO4J_PASS", "Manami1008"))
)

print("▶ Loading SciBERT …")
sci_model = SentenceTransformer("allenai/scibert_scivocab_uncased")
sci_model.eval()
embed = lambda txt: sci_model.encode(
    txt, convert_to_numpy=True, normalize_embeddings=True
).tolist()

# ─────────── ANN : text embedding branch ───────────────────────
def vector_topN(qvec, n=TOP_N):
    with driver.session() as s:
        rows = s.run(
            """
            CALL db.index.vector.queryNodes('paper_vec', $n, $q)
            YIELD node, score
            RETURN node.id AS id, score
            ORDER BY score
            """,
            q=qvec, n=n
        ).data()
    return [{"id": r["id"], "rank_vec": i}           # store rank (0-based)
            for i, r in enumerate(rows)]

# ─────────── Graph branch (Topic / FoS coverage) ───────────────
# ─────────── Graph branch (Topic / FoS coverage) ───────────────
# Improved multi-stage concept → Topic/FoS matcher
#   • stage-1: high-confidence ANN  (≥ SIM_HI)
#   • stage-2: lower-confidence ANN (≥ SIM_LO, take TOP_K_LO per concept)
#   • stage-3: lexical fallback     (substring match on labels)

SIM_HI      = 0.50          # high-confidence threshold
SIM_LO      = 0.30          # lower-confidence threshold
TOP_K_LO    = 2             # neighbours per concept in stage-2
MIN_HITS    = 3             # stop once we have this many IDs

def ann_nearest(kind: str, vec, k: int):
    """kind ∈ {'topic','fos'}  – return list[(idx, sim)] length ≤ k."""
    index = "topic_vec" if kind == "topic" else "fos_vec"
    q = f"""
        CALL db.index.vector.queryNodes('{index}', $k, $v)
        YIELD node, score
        RETURN node.idx AS idx, score
    """
    rows = driver.session().run(q, k=k, v=vec).data()
    return [(int(r["idx"]), 1 - r["score"]) for r in rows]

def lexical_candidates(term: str):
    """Fallback substring match against KG labels – returns two ID sets."""
    term_lc = term.lower()
    q = """
    MATCH (x)
    WHERE   (x:Topic        AND toLower(x.keywords) CONTAINS $t)
        OR  (x:FieldOfStudy AND toLower(x.name)     CONTAINS $t)
    RETURN labels(x)[0] AS kind, x.idx AS idx
    LIMIT 10
    """
    out_topic, out_fos = set(), set()
    for r in driver.session().run(q, t=term_lc):
        if r["kind"] == "Topic":
            out_topic.add(int(r["idx"]))
        else:
            out_fos.add(int(r["idx"]))
    return out_topic, out_fos

def build_id_lists(concepts):
    topic_ids, fos_ids = set(), set()

    # ── Stage 1: high-confidence ANN ──
    for c in concepts:
        if not isinstance(c, str) or len(c.split()) < 2:
            continue
        vec = embed(c)
        best = ann_nearest("topic", vec, 1) + ann_nearest("fos", vec, 1)
        if best:
            idx, sim = max(best, key=lambda t: t[1])
            if sim >= SIM_HI:
                if idx == best[0][0]:
                    topic_ids.add(idx)
                else:
                    fos_ids.add(idx)

    if len(topic_ids) + len(fos_ids) >= MIN_HITS:
        return list(topic_ids), list(fos_ids)

    # ── Stage 2: lower-confidence ANN ──
    for c in concepts:
        vec = embed(c)
        for idx, sim in ann_nearest("topic", vec, TOP_K_LO):
            if sim >= SIM_LO:
                topic_ids.add(idx)
        for idx, sim in ann_nearest("fos", vec, TOP_K_LO):
            if sim >= SIM_LO:
                fos_ids.add(idx)

    if len(topic_ids) + len(fos_ids) >= MIN_HITS:
        return list(topic_ids), list(fos_ids)

    # ── Stage 3: lexical fallback ──
    for c in concepts:
        t_ids, f_ids = lexical_candidates(c)
        topic_ids.update(t_ids)
        fos_ids.update(f_ids)

    return list(topic_ids), list(fos_ids)
# ─────────── end of matcher replacement ───────────────────────

CYPHER_GRAPH = """
MATCH (p:Paper)
OPTIONAL MATCH (p)-[:HAS_TOPIC]->(t:Topic) WHERE t.idx IN $topicIds
OPTIONAL MATCH (p)-[:HAS_FOS]->(f:FieldOfStudy) WHERE f.idx IN $fosIds
WITH p, count(DISTINCT t)+count(DISTINCT f) AS hits
WHERE hits>0
RETURN p.id AS id, hits
ORDER BY hits DESC, p.year DESC
LIMIT $n
"""

def graph_topN(topicIds, fosIds, n=TOP_N):
    if not topicIds and not fosIds:
        return []
    with driver.session() as s:
        rows = s.run(
            CYPHER_GRAPH, topicIds=topicIds, fosIds=fosIds, n=n
        ).data()
    return [{"id": r["id"], "rank_graph": i} for i, r in enumerate(rows)]

# ─────────── LLaMA concept extractor ───────────────────────────
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
print("▶ Loading LLaMA-8B …")
llm = pipeline(
    "text-generation",
    model=AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.float16),
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID)
)

# escape braces inside the JSON template ({{ }})
PROMPT = (
    "You are an academic assistant.\n"
    "Extract 3–7 salient research concepts from <TEXT> and return JSON in the form:\n"
    "{{\"concepts\": [\"...\"]}}\n"
    "<TEXT>{paragraph}</TEXT>"
)

def llm_extract(paragraph: str):
    query = PROMPT.format(paragraph=paragraph.strip())     # now safe
    out = llm(query, max_new_tokens=128,
              temperature=0.0, do_sample=False)[0]["generated_text"]
    for chunk in re.findall(r"\{[^{}]+\}", out, re.S):
        for loader in (json.loads, ast.literal_eval):
            try:
                d = loader(chunk)
                if "concepts" in d:
                    return d
            except Exception:
                pass
    raise ValueError("LLM output unparsable:\n" + out)



# ─────────── RRF Fusion ────────────────────────────────────────
def reciprocal_rank(rank:int, k:int=RRF_K)->float:
    """Convert 0-based rank to reciprocal-rank score."""
    return 1.0 / (k + rank)

def fuse_rrf(vec_rows, graph_rows):
    pool = {}
    for row in vec_rows:
        pool.setdefault(row["id"], 0.0)
        pool[row["id"]] += reciprocal_rank(row["rank_vec"])
    for row in graph_rows:
        pool.setdefault(row["id"], 0.0)
        pool[row["id"]] += reciprocal_rank(row["rank_graph"])
    ranked = sorted(pool.items(), key=lambda x: x[1], reverse=True)
    return ranked   # list of (id, fused_score)

# ─────────── Utilities for explanations / summaries ────────────
bert_tok = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

def make_whitelist_tokens(texts, extra_terms):
    vocab=set()
    for txt in texts+extra_terms:
        vocab.update(bert_tok.tokenize(txt))
    vocab.update([",",".","(",")","the","of","and","to","is","in","for"])
    return {bert_tok.convert_tokens_to_ids(t) for t in vocab if t in bert_tok.vocab}

def shortest_path_sentence(pid, topicIds, fosIds):
    q="""
    MATCH (p:Paper {id:$pid}),(x)
    WHERE (x:Topic AND x.idx IN $topicIds) OR (x:FieldOfStudy AND x.idx IN $fosIds)
    WITH p,x LIMIT 1
    MATCH pth=shortestPath((p)-[*..3]-(x))
    RETURN [n IN nodes(pth)|labels(n)[0]] AS labs,
           [n IN nodes(pth)|coalesce(n.title,n.keywords,n.name,n.idx)] AS names
    """
    rec=driver.session().run(q,pid=pid,topicIds=topicIds,fosIds=fosIds).single()
    if not rec: return ""
    labs,names=rec["labs"],rec["names"]
    hops=" → ".join(f"{names[i]} ({labs[i]})" for i in range(len(names)))
    return f"Path: {hops}"

def safe_summary(title, abstract, paragraph, whitelist_ids, timeout=2.0):
    prompt=f"""Context:
TITLE: {title}
ABSTRACT: {abstract}

Paragraph: {paragraph}

Write two sentences explaining why this paper is relevant. Use ONLY words from the context."""
    try:
        out=llm(prompt,max_new_tokens=60,temperature=0.0,do_sample=False,
                timeout=timeout)[0]["generated_text"]
        for tok in out.split():
            if bert_tok.convert_tokens_to_ids(tok) not in whitelist_ids:
                raise ValueError
        return out.strip()
    except Exception:
        return ". ".join(abstract.split(". ")[:2]).strip()

# ─────────── Main CLI ──────────────────────────────────────────
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("-p","--paragraph",required=True)
    args=ap.parse_args()

    # 1) Concept extraction
    info=llm_extract(args.paragraph)
    print("\nLLM concepts:", info["concepts"])

    topicIds,fosIds=build_id_lists(info["concepts"])

    # 2) Parallel retrieval
    vec_rows   = vector_topN(embed(args.paragraph))
    graph_rows = graph_topN(topicIds,fosIds)

    # 3) Rank-based fusion
    fused = fuse_rrf(vec_rows, graph_rows)          # list[(id,score)]

    # 4) Gather metadata and explanations
    top_ids = [doc_id for doc_id,_ in fused[:10]]
    with driver.session() as s:
        meta=s.run(
            "MATCH (p:Paper) WHERE p.id IN $ids "
            "RETURN p.id AS id, p.title AS title, p.year AS year, p.abstract AS abs",
            ids=top_ids).data()
    meta_d = {m["id"]:m for m in meta}

    rows=[]
    for rank,(pid,score) in enumerate(fused[:10]):
        m=meta_d.get(pid,{"title":"(missing)","year":"","abs":""})
        path=shortest_path_sentence(pid,topicIds,fosIds)
        if rank<3:
            whitelist=make_whitelist_tokens([m["title"],m["abs"]],info["concepts"])
            summ=safe_summary(m["title"],m["abs"],args.paragraph,whitelist)
        else:
            summ=". ".join(m["abs"].split(". ")[:2]).strip() or "(abstract unavailable)"
        rows.append(dict(id=pid,title=m["title"],year=m["year"],
                         score=round(score,4),
                         summary=summ,explain_path=path))

    df=pd.DataFrame(rows)
    print("\nTop-10 ranked papers (RRF fusion):\n")
    print(df.to_markdown(index=False))


# python best_recommend.py -p "Compared to desktop computing, designing hardware and software for mobile computing presents a host of unique challenges, particularly because location, environment, connectivity, and other important factors are commonly unpredictable and dynamic. The strategies that have been demonstrated to be effective for desktop computing are only minimally useful for mobile computing. Clearly, different design and evaluation paradigms need to exist for mobile computing devices and environments. One study cites the inadequacy of the desktop metaphor for mobile computing for information presentation. This is merely a single example of the dissonance between effective desktop and mobile computing strategies. Another source notes that human-computer interaction has developed a good understanding of how to design and evaluate forms of interaction in fixed contexts of use, but this is not the situation for mobile computing. This highlights the issue of differences between desktop and mobile computing in terms of contexts of use. It has been pointed out that for traditional desktop computing applications, tasks take place within the computer, while for mobile computing, tasks typically reside outside of the computer, such as navigation or data recording. Thus, in many mobile computing interactions, there are multiple tasks taking place, often with the mobile task being secondary, which is why the context of use must be considered."

# python best_recommend.py -p "The key to improving the performance of parallel sparse LU factorization on second-class message passing platforms is to reduce inter-processor synchronization granularity and communication volume. In this paper, we examine the message passing overhead in parallel sparse LU factorization with two-dimensional data mapping and investigate techniques to reduce such overhead. Our main finding is that this objective can be achieved with a small amount of extra computation and slightly weakened numerical stability. Although such trade-offs may not be worthwhile on systems with high message passing performance, these techniques can be very beneficial for second-class message passing platforms. In particular, we propose a novel technique called speculative batch pivoting, in which large elements for a group of columns across all processors are collected at one processor, and the pivot selections for these columns are made together through speculative factorization. These pivot selections are accepted if the chosen pivots pass a numerical stability test; otherwise, the scheme falls back to the conventional column-by-column pivot selection for this group of columns. Speculative batch pivoting substantially decreases the inter-processor synchronization granularity compared with the conventional approach. This reduction is achieved at the cost of increased computation, specifically the cost of speculative factorization."
# python explainability.py -p "In image recognition, VLAD is a representation that encodes by the residual vectors with respect to a dictionary, and Fisher Vector [30] can be formulated as a probabilistic version [18] of VLAD. Both of them are powerful shallow representations for image retrieval and classification [4, 48]. For vector quantization, encoding residual vectors [17] is shown to be more effective than encoding original vectors. In low-level vision and computer graphics, for solving Partial Differential Equations (PDEs), the widely used Multigrid method [3] reformulates the system as subproblems at multiple scales, where each subproblem is responsible for the residual solution between a coarser and a finer scale. An alternative to Multigrid is hierarchical basis preconditioning [45, 46], which relies on variables that represent residual vectors between two scales. It has been shown [3, 45, 46] that these solvers converge much faster than standard solvers that are unaware of the residual nature of the solutions. These methods suggest that a good reformulation or preconditioning can simplify the optimization."




