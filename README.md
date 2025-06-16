Paper:
https://www.overleaf.com/project/68317ee77bc2d9cfe0cf412e

Ideas:
https://docs.google.com/document/d/1w6oR8HBcd2USkIyLxnu0ILzq_3Q6FVjpHXXyZW3aknA/edit?hl=JA&tab=t.0

1) Sliding window chunking based on token/sentense
2) BM25 (TFiDF base) and embeding search -> cancdidate retrieval
3) 2 hop reasoning over citation edges and keywords and topics
4) reranking based on llm 

Current Best Results 
| Metric       | LLaMA CoT + CARS | BM25 Full Text |
| ------------ | ---------------- | -------------- |
| **P\@3**     | 0.3200           | 0.2533         |
| **HR\@3**    | 0.6000           | 0.5600         |
| **R\@3**     | 0.0337           | 0.0262         |
| **NDCG\@3**  | 0.3477           | 0.4400         |
| **P\@5**     | 0.2960           | 0.2000         |
| **HR\@5**    | 0.6800           | 0.6800         |
| **R\@5**     | 0.0520           | 0.0342         |
| **NDCG\@5**  | 0.3231           | 0.4871         |
| **P\@10**    | 0.2440           | 0.1840         |
| **HR\@10**   | 0.8800           | 0.7600         |
| **R\@10**    | 0.0875           | 0.0656         |
| **NDCG\@10** | 0.2776           | 0.5025         |
| **P\@15**    | 0.2213           | 0.1520         |
| **HR\@15**   | 0.9600           | 0.7600         |
| **R\@15**    | 0.1174           | 0.0791         |
| **NDCG\@15** | 0.2552           | 0.4935         |
| **P\@20**    | 0.2120           | 0.1320         |
| **HR\@20**   | 0.9600           | 0.8000         |
| **R\@20**    | 0.1486           | 0.0906         |
| **NDCG\@20** | 0.2499           | 0.4948         |
