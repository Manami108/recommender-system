## Draft for Conference
https://www.overleaf.com/project/68317ee77bc2d9cfe0cf412e

## Draft (Word version)
https://docs.google.com/document/d/1oCDwL9tWEbsL7TUPOpL7cG2EbPaNwulRCVhUCcJNR54/edit?usp=sharing

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
| **P@3**      | 0.3867          | 0.2533         | 0.1600     | 0.3600      |
| **HR@3**     | 0.6800          | 0.5600         | 0.2800     | 0.6000      |
| **R@3**      | 0.0420          | 0.0262         | 0.0135     | 0.0369      |
| **NDCG@3**   | 0.4163          | 0.4400         | 0.2452     | 0.3837      |
| **P@5**      | 0.3520          | 0.2000         | 0.1520     | 0.3520      |
| **HR@5**     | 0.7600          | 0.6800         | 0.4800     | 0.7200      |
| **R@5**      | 0.0614          | 0.0342         | 0.0222     | 0.0612      |
| **NDCG@5**   | 0.3832          | 0.4871         | 0.3283     | 0.3725      |
| **P@10**     | 0.3040          | 0.1840         | 0.1600     | 0.2680      |
| **HR@10**    | 0.8400          | 0.7600         | 0.7200     | 0.8000      |
| **R@10**     | 0.1067          | 0.0656         | 0.0525     | 0.0919      |
| **NDCG@10**  | 0.3392          | 0.5025         | 0.3905     | 0.3077      |
| **P@15**     | 0.2400          | 0.1520         | 0.1440     | 0.2213      |
| **HR@15**    | 0.8400          | 0.7600         | 0.8000     | 0.8800      |
| **R@15**     | 0.1277          | 0.0791         | 0.0754     | 0.1150      |
| **NDCG@15**  | 0.2883          | 0.4935         | 0.4173     | 0.2675      |
| **P@20**     | 0.2140          | 0.1320         | 0.1320     | 0.1980      |
| **HR@20**    | 0.8400          | 0.8000         | 0.8000     | 0.8800      |
| **R@20**     | 0.1501          | 0.0906         | 0.0906     | 0.1402      |
| **NDCG@20**  | 0.2701          | 0.4948         | 0.4192     | 0.2503      |


Car's zero
P@3        0.3733
HR@3       0.6400
R@3        0.0401
NDCG@3     0.3669
P@5        0.3600
HR@5       0.7600
R@5        0.0661
NDCG@5     0.3592
P@10       0.3080
HR@10      0.8800
R@10       0.1096
NDCG@10    0.3234
P@15       0.2667
HR@15      0.8800
R@15       0.1402
NDCG@15    0.2918
P@20       0.2360
HR@20      0.8800
R@20       0.1661
NDCG@20    0.2739
