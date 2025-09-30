# INSPIRE — Context-Aware Scholarly Paper Recommendation (Overview)

**INSPIRE** treats a **draft paragraph** as a real-time signal of **evolving research intent**.  
It combines **hybrid retrieval** (BM25 + SciBERT + RRF), **multi-hop reasoning** on a heterogeneous **knowledge graph (KG)**, and a lightweight **LLM reranker** to surface both directly relevant and *bridge* papers.

## Key Ideas
- **Writing-as-Thinking:** draft text ≈ evolving intent (planning ↔ translating ↔ reviewing).
- **Hybrid Retrieval:** four sources → full-paragraph BM25, full-paragraph embedding (SciBERT), chunk BM25, chunk embedding → fused by **RRF (k=60)**.
- **KG Reasoning:** one-hop **topic/FoS** and **citation** expansions; simple weighting (topic/FoS co-neighbor + citation indicator) to form expanded set \(C'\).
- **LLM Reranking:** ranks \(C'\) by contextual fit to the draft; candidates remain grounded in the corpus (no generation of new papers).

Data Prep (Brief)
- Corpus: DBLP v12 (papers + citations).
- Embeddings: SciBERT on title+abstract.
- Topics: BERTopic → one topic label / paper.
- KG: Neo4j with nodes: paper, topic, fos; edges: cites, has_topic, has_fos.

To be continue... (It will be updated soon!)
