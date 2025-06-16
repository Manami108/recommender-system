Paper:
https://www.overleaf.com/project/68317ee77bc2d9cfe0cf412e

Ideas:
https://docs.google.com/document/d/1w6oR8HBcd2USkIyLxnu0ILzq_3Q6FVjpHXXyZW3aknA/edit?hl=JA&tab=t.0

1) Sliding window chunking based on token/sentense
2) BM25 (TFiDF base) and embeding search -> cancdidate retrieval
3) 2 hop reasoning over citation edges and keywords and topics
4) reranking based on llm 

Current Best Results 
Evaluation with LLaMA Zero-shot Chain of Thought CARS prompting 
 P@3        0.3200
HR@3       0.6000
R@3        0.0337
NDCG@3     0.3477
P@5        0.2960
HR@5       0.6800
R@5        0.0520
NDCG@5     0.3231
P@10       0.2440
HR@10      0.8800
R@10       0.0875
NDCG@10    0.2776
P@15       0.2213
HR@15      0.9600
R@15       0.1174
NDCG@15    0.2552
P@20       0.2120
HR@20      0.9600
R@20       0.1486
NDCG@20    0.2499
