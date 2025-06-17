## Draft for Conference
https://www.overleaf.com/project/68317ee77bc2d9cfe0cf412e

## Ideas
https://docs.google.com/document/d/1w6oR8HBcd2USkIyLxnu0ILzq_3Q6FVjpHXXyZW3aknA/edit?hl=JA&tab=t.0

## Experimental Protocol
https://docs.google.com/document/d/10WN7Q7yhme27JXIMy7q-j8Hbx9Amyt1D6Aol2LYlTvo/edit?tab=t.0

## Approach
1) Sliding window chunking based on token
2) BM25 (TFiDF-like back of words base) and embeding search -> Reciprocal Rank Fusion (RRF scoring) -> cancdidate retrievals
3) 2 hop reasoning over citation edges and keywords and topics (should i do that?)
4) reranking based on llm
5) Precision, Recall, Hit Rate, NDCG based evaluations

## Current Best Results

| Metric       | Zero CoT + CARS | BM25 Full Text | Hybrid RRF | Pure LLaMA  |
| ------------ | --------------- | -------------- | -----------| ----------- |
| **P@3**      | 0.3600          | 0.2533         | 0.1600     | 0.3600      |
| **HR@3**     | 0.7200          | 0.5600         | 0.2800     | 0.6000      |
| **R@3**      | 0.0392          | 0.0262         | 0.0135     | 0.0369      |
| **NDCG@3**   | 0.3861          | 0.4400         | 0.2452     | 0.3837      |
| **P@5**      | 0.3280          | 0.2000         | 0.1520     | 0.3520      |
| **HR@5**     | 0.7600          | 0.6800         | 0.4800     | 0.7200      |
| **R@5**      | 0.0588          | 0.0342         | 0.0222     | 0.0612      |
| **NDCG@5**   | 0.3579          | 0.4871         | 0.3283     | 0.3725      |
| **P@10**     | 0.2800          | 0.1840         | 0.1600     | 0.2680      |
| **HR@10**    | 0.9200          | 0.7600         | 0.7200     | 0.8000      |
| **R@10**     | 0.1038          | 0.0656         | 0.0525     | 0.0919      |
| **NDCG@10**  | 0.3141          | 0.5025         | 0.3905     | 0.3077      |
| **P@15**     | 0.2453          | 0.1520         | 0.1440     | 0.2213      |
| **HR@15**    | 0.9200          | 0.7600         | 0.8000     | 0.8800      |
| **R@15**     | 0.1328          | 0.0791         | 0.0754     | 0.1150      |
| **NDCG@15**  | 0.2829          | 0.4935         | 0.4173     | 0.2675      |
| **P@20**     | 0.2180          | 0.1320         | 0.1320     | 0.1980      |
| **HR@20**    | 0.9200          | 0.8000         | 0.8000     | 0.8800      |
| **R@20**     | 0.1552          | 0.0906         | 0.0906     | 0.1402      |
| **NDCG@20**  | 0.2658          | 0.4948         | 0.4192     | 0.2503      |
