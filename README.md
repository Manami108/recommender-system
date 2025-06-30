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

