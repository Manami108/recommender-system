import os
import pandas as pd
from neo4j import GraphDatabase

#──────────────────────────────────────────────────────────────────────────────
# Neo4j connection (reuse existing driver settings)
#──────────────────────────────────────────────────────────────────────────────
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASS", "Manami1008")
    )
)

def multi_hop_topic_citation_reasoning(
    pids: list[str],
    max_topic_hops: int = 2,
    top_n: int = 50
) -> pd.DataFrame:
    """
    Hybrid multi-hop reasoning over Topic/FoS shared nodes and citation edges.

    - Topic/FoS hops:
      * Hop 1: seed → Topic/FoS → other papers
      * Hop 2: seed → Topic/FoS → intermediate → Topic/FoS → other papers
    - Citation hop (one hop in either direction): seed ↔ CITES ↔ other papers

    Returns a DataFrame with columns:
        pid, title, year, hop (1 or 2), shared_count, src ('topic' or 'cite')
    """
    if not pids:
        return pd.DataFrame(columns=["pid","title","year","hop","shared_count","src"])

    results = []
    with driver.session() as session:
        # Topic/FoS Hop 1
        topic1 = session.run(
            '''
            UNWIND $pids AS seed_id
            MATCH (seed:Paper {id: seed_id})-[:HAS_TOPIC|:HAS_FOS]->(t)<-[:HAS_TOPIC|:HAS_FOS]-(other:Paper)
            WHERE other.id <> seed_id AND NOT other.id IN $pids
            WITH other, COUNT(DISTINCT t) AS shared_count
            RETURN other.id AS pid,
                   other.title AS title,
                   other.year AS year,
                   1 AS hop,
                   shared_count,
                   'topic' AS src
            ORDER BY shared_count DESC, year DESC
            LIMIT $top_n
            '''
        , pids=pids, top_n=top_n).data()
        results += topic1

        # Citation hop
        cite = session.run(
            '''
            UNWIND $pids AS seed_id
            OPTIONAL MATCH (seed:Paper {id: seed_id})-[:CITES]->(cited:Paper)
            OPTIONAL MATCH (citing:Paper)-[:CITES]->(seed)
            WITH collect(DISTINCT cited) + collect(DISTINCT citing) AS cand
            UNWIND cand AS other
            WHERE other.id IS NOT NULL AND NOT other.id IN $pids
            RETURN other.id AS pid,
                   other.title AS title,
                   other.year AS year,
                   1 AS hop,
                   1 AS shared_count,
                   'cite' AS src
            ORDER BY year DESC
            LIMIT $top_n
            '''
        , pids=pids, top_n=top_n).data()
        results += cite

        # Topic/FoS Hop 2
        if max_topic_hops >= 2 and topic1:
            hop1_ids = [r['pid'] for r in topic1]
            topic2 = session.run(
                '''
                UNWIND $hop1_ids AS mid_id
                MATCH (mid:Paper {id: mid_id})-[:HAS_TOPIC|:HAS_FOS]->(t2)<-[:HAS_TOPIC|:HAS_FOS]-(other:Paper)
                WHERE other.id <> mid_id AND NOT other.id IN $pids
                WITH other, COUNT(DISTINCT t2) AS shared_count
                RETURN other.id AS pid,
                       other.title AS title,
                       other.year AS year,
                       2 AS hop,
                       shared_count,
                       'topic' AS src
                ORDER BY shared_count DESC, year DESC
                LIMIT $top_n
                '''
            , hop1_ids=hop1_ids, pids=pids, top_n=top_n).data()
            results += topic2

    df = pd.DataFrame(results)
    if df.empty:
        return df

    # Keep first occurrence per pid, sorted by hop then by shared_count
    df = (
        df.sort_values(['hop','shared_count'], ascending=[True, False])
          .drop_duplicates('pid')
          .reset_index(drop=True)
    )
    return df

# Example usage:
# from hop_reasoning import multi_hop_topic_citation_reasoning
# seeds = ['paper1', 'paper2']
# df = multi_hop_topic_citation_reasoning(seeds)
# print(df)
