## Draft for Conference
https://www.overleaf.com/project/68317ee77bc2d9cfe0cf412e

## Ideas
https://docs.google.com/document/d/1w6oR8HBcd2USkIyLxnu0ILzq_3Q6FVjpHXXyZW3aknA/edit?hl=JA&tab=t.0

## Approach
1) Sliding window chunking based on token
2) BM25 (TFiDF-like back of words base) and embeding search -> Reciprocal Rank Fusion (RRF scoring) -> cancdidate retrievals
3) 2 hop reasoning over citation edges and keywords and topics (should i do that?)
4) reranking based on llm
5) Precision, Recall, Hit Rate, NDCG based evaluations

## Current Best Results
| Metric       | Zero CoT + CARS | BM25 Full Text | Hybrid RRF |
| ------------ | --------------- | -------------- | ----------- |
| **P@3**      | 0.3200          | 0.2533         | 0.1600      |
| **HR@3**     | 0.6000          | 0.5600         | 0.2800      |
| **R@3**      | 0.0337          | 0.0262         | 0.0135      |
| **NDCG@3**   | 0.3477          | 0.4400         | 0.2452      |
| **P@5**      | 0.2960          | 0.2000         | 0.1520      |
| **HR@5**     | 0.6800          | 0.6800         | 0.4800      |
| **R@5**      | 0.0520          | 0.0342         | 0.0222      |
| **NDCG@5**   | 0.3231          | 0.4871         | 0.3283      |
| **P@10**     | 0.2440          | 0.1840         | 0.1600      |
| **HR@10**    | 0.8800          | 0.7600         | 0.7200      |
| **R@10**     | 0.0875          | 0.0656         | 0.0525      |
| **NDCG@10**  | 0.2776          | 0.5025         | 0.3905      |
| **P@15**     | 0.2213          | 0.1520         | 0.1440      |
| **HR@15**    | 0.9600          | 0.7600         | 0.8000      |
| **R@15**     | 0.1174          | 0.0791         | 0.0754      |
| **NDCG@15**  | 0.2552          | 0.4935         | 0.4173      |
| **P@20**     | 0.2120          | 0.1320         | 0.1320      |
| **HR@20**    | 0.9600          | 0.8000         | 0.8000      |
| **R@20**     | 0.1486          | 0.0906         | 0.0906      |
| **NDCG@20**  | 0.2499          | 0.4948         | 0.4192      |
