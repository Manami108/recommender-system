# inspect_topics.py
from neo4j import GraphDatabase
import os

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER","neo4j"),
          os.getenv("NEO4J_PASS","Manami1008"))
)

with driver.session() as session:
    # Get some Topic nodes
    tops = session.run(
        "MATCH (t:Topic) RETURN t.name AS name LIMIT 50"
    ).data()
    print("=== Sample Topics ===")
    for r in tops:
        print("•", r["name"])

    # Get some FoS nodes
    foss = session.run(
        "MATCH (f:FieldOfStudy) RETURN f.name AS name LIMIT 50"
    ).data()
    print("\n=== Sample Fields of Study (FoS) ===")
    for r in foss:
        print("•", r["name"])
