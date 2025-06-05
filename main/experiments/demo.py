
# demo.py
from chain_of_thought import extract_context
from recall_bm25     import recall_candidates
from rerank_llm      import llm_rerank

paragraph = """This paper accordingly proposes a novel Context-guided Triple Matching (CTM), while the third component missing from the pairwise matching is adopted as a prior context. The proposed triple matching is present as a hierarchical attention flow to adequately capture the semantic relationship. Specifically, given a candidate triple, we first employ (any) one component from the triple as the prior context. Then we apply the bidirectional attention to calculate the correlation between context and the other two components separately. Afterwards, another attention layer is utilized to leverage two above correlations to form an aggregated context-aware representation. In this way, the model is able to gather more comprehensive semantic relationship for the triple, according to the selected context. Similarly, we enumerate the other two components (from the triple) and cast as the prior context to repeat the same attention flow. Finally, a fully-connected layer is employed for all formed context-aware representations to estimate the matching score. In addition to the triple matching, we also consider to adopt a contrastive regularization in capturing the subtle semantic differences among answer candidates. The aim is to maximize the similarity of features from correct triple(s) while pushing away that of distractive ones, that has been neglected by existing methods."""

ctx  = extract_context(paragraph)
print("\nContext:", ctx, "\n")

cand = recall_candidates(ctx, limit=60)
best = llm_rerank(paragraph, cand, k=15)

print(best[["pid","llm_score","title"]].to_string(index=False))
