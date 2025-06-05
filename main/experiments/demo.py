# demo.py

from chain_of_thought import extract_context
from recall_bm25     import recall_candidates
from rerank_llm      import llm_rerank

paragraph = """
This paper accordingly proposes a novel Context-guided Triple Matching (CTM),
while the third component missing from the pairwise matching is adopted as a prior context.
The proposed triple matching is present as a hierarchical attention flow to adequately capture
the semantic relationship. Specifically, given a candidate triple, we first employ (any) one
component from the triple as the prior context. Then we apply the bidirectional attention to
calculate the correlation between context and the other two components separately. Afterwards,
another attention layer is utilized to leverage two above correlations to form an aggregated
context-aware representation. In this way, the model is able to gather more comprehensive
semantic relationship for the triple, according to the selected context. Similarly, we enumerate
the other two components (from the triple) and cast as the prior context to repeat the same
attention flow. Finally, a fully-connected layer is employed for all formed context-aware
representations to estimate the matching score. In addition to the triple matching, we also
consider to adopt a contrastive regularization in capturing the subtle semantic differences among
answer candidates. The aim is to maximize the similarity of features from correct triple(s) while
pushing away that of distractive ones, that has been neglected by existing methods.
"""

# ─── Stage 1: Context Extraction ───────────────────────────────────────
ctx = extract_context(paragraph)
print("\n— Extracted context signals —")
for k, v in ctx.items():
    print(f"{k:<18}: {v}")

# ─── Stage 2a: Recall (role-AND + BM25 fallback) ───────────────────────
candidates_df = recall_candidates(ctx, limit=60)
print(f"\nFound {len(candidates_df)} candidate papers.")

# ─── Stage 2b: LLM Rerank ──────────────────────────────────────────────
if candidates_df.empty:
    print("No candidates to rerank (empty recall set).")
else:
    best = llm_rerank(paragraph, candidates_df, k=15)
    if best.empty:
        print("LLM reranker produced an empty top-K (no applicable candidates).")
    else:
        print("\nTop recommendations after LLM reranking:\n")
        print(best[["pid", "title", "llm_score"]].to_string(index=False))
