# eval_utils.py
import numpy as np
from typing import Optional
from neo4j import GraphDatabase

TOPK_LIST = (3,5,10,15,20)

def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T

def precision_by_sim(pred, sim_map, k):
    return sum(sim_map.get(p,False) for p in pred[:k]) / k if k else 0.0

def hit_rate_by_sim(pred, sim_map, k):
    return float(any(sim_map.get(p,False) for p in pred[:k]))

def recall_by_sim(pred, sim_map, k, n_rel):
    if n_rel==0: return 0.0
    return sum(sim_map.get(p,False) for p in pred[:k]) / n_rel

def ndcg_by_sim(pred, sim_map, k):
    rels = [1 if sim_map.get(p,False) else 0 for p in pred[:k]]
    dcg   = sum(r/np.log2(i+2) for i,r in enumerate(rels))
    ideal=sorted(rels,reverse=True)
    idcg  = sum(r/np.log2(i+2) for i,r in enumerate(ideal))
    return dcg/idcg if idcg>0 else 0.0

def fetch_year_by_doi(doi: Optional[str]) -> Optional[int]:
    if not doi: return None
    # copy your Neo4j URI/auth into this module or read from env
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"," bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER","neo4j"),os.getenv("NEO4J_PASS","Manami1008"))
    )
    q="MATCH (p:Paper {doi:$doi}) RETURN p.year AS year"
    with driver.session() as sess:
        r=sess.run(q,doi=doi).single()
    return r["year"] if r and r["year"] else None
