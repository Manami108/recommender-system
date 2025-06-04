Paper:
https://www.overleaf.com/project/68317ee77bc2d9cfe0cf412e

Ideas:
https://docs.google.com/document/d/1w6oR8HBcd2USkIyLxnu0ILzq_3Q6FVjpHXXyZW3aknA/edit?hl=JA&tab=t.0

1. Overall Two‐Stage Architecture
1) Stage 1: Context Extraction (LLM as “Interpreter”)
   Input: a user’s paragraph (e.g., draft of an introduction).
   Output: a structured representation of “context signals” (e.g., main topic, subtopics, problem statement, technologies, domain, intent).
2) Stage 2: Graph‐Based Retrieval & Reranking
   Input: structured context signals + (optionally) paragraph embedding.
   Process: Use context signals to issue constrained graph queries (FoS, topic‐node matching, citation‐traversal).
            Retrieve a candidate set of papers (e.g., top N from each subquery).
            Rerank candidates using LLM‐driven scoring (but only scoring, never generating new content).
   Output: a top‐K list of KG‐verified papers.

Splitting the system into “LLM→extract context” and “KG→retrieve” ensures (a) context understanding and (b) no hallucinated citations, since all recommended paper IDs come from your KG.
