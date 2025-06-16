Paper:
https://www.overleaf.com/project/68317ee77bc2d9cfe0cf412e

Ideas:
https://docs.google.com/document/d/1w6oR8HBcd2USkIyLxnu0ILzq_3Q6FVjpHXXyZW3aknA/edit?hl=JA&tab=t.0

1) Sliding window chunking based on token/sentense
2) BM25 (TFiDF base) and embeding search -> cancdidate retrieval
3) 2 hop reasoning over citation edges and keywords and topics
4) reranking based on llm 

Current Best Results 
Evaluation with LLaMA Zero-shot Chain of Thought CARS prompting <br>
P@3        0.3200  <br>
HR@3       0.6000  <br>
R@3        0.0337  <br>
NDCG@3     0.3477  <br>
P@5        0.2960  <br>
HR@5       0.6800  <br>
R@5        0.0520  <br>
NDCG@5     0.3231  <br>
P@10       0.2440  <br>
HR@10      0.8800  <br>
R@10       0.0875  <br>
NDCG@10    0.2776  <br>
P@15       0.2213  <br>
HR@15      0.9600  <br>
R@15       0.1174  <br>
NDCG@15    0.2552  <br>
P@20       0.2120  <br>
HR@20      0.9600  <br>
R@20       0.1486  <br>
NDCG@20    0.2499  <br>
