import os
import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from chunking import clean_text, chunk_tokens, chunk_sentences
from recall import recall_by_chunks, expand_citation_hops, fetch_metadata

# ─────────────────────────────────────────────────────────────────────────────
# Demo script: candidate retrieval pipeline only
# 1) Clean & chunk paragraph
# 2) Combined recall via BM25 & embedding
# 3) Citation expansion (2-hop)
# 4) Aggregate & dedupe final candidate list
# 5) Output results (no reranking)
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Example user paragraph (replace with real input)
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

    # 2) Clean & chunk the paragraph
    cleaned = clean_text(paragraph)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    # choose token-level or sentence-level chunks
    chunks = chunk_tokens(cleaned, tokenizer, win=128, stride=64)
    # alternatively: chunks = chunk_sentences(cleaned, n_sent=3, stride=1)

    print(f"Generated {len(chunks)} chunks:")
    for i, c in enumerate(chunks, 1):
        print(f"  {i}. {c[:80].strip()}...")
    print()

    # 3) Combined recall via BM25 & embedding
    rec_df = recall_by_chunks(chunks, k_bm25=40, k_vec=40, sim_th=0.30)
    print(f"Retrieved {len(rec_df)} unique candidates from recall stage.")
    print(rec_df[['pid','title','sim','src']].head(20).to_string(index=False))
    print()

    # 4) 2-hop citation expansion
    seed_pids = rec_df['pid'].tolist()
    cit_df = expand_citation_hops(seed_pids, max_hops=2, limit_per_hop=100)
    print(f"Expanded to {len(cit_df)} citation-hop candidates.")
    print(cit_df[['pid','title','hop']].head(20).to_string(index=False))
    print()

    # 5) Merge recall + citation
    combined = pd.concat([
        rec_df.assign(source='recall'),
        cit_df.assign(source='citation', sim=np.nan)
    ], ignore_index=True)
    # dedupe: prefer lowest hop then highest similarity
    combined = combined.sort_values(['hop','sim'], ascending=[True, False])
    combined = combined.drop_duplicates('pid', keep='first').reset_index(drop=True)
    print(f"Total {len(combined)} candidates after merging.")
    print()

    # 6) Optional: fetch additional metadata
    meta = fetch_metadata(combined['pid'].tolist())
    # keep only pid, abstract, authors to avoid overwriting title/year
    meta = meta[['pid','abstract','authors']]
    final_df = combined.merge(meta, on='pid', how='left')

    # 7) Display final list
    print("\nFinal candidate list:")
    print(final_df[['pid','title','year','source','hop','sim']].to_string(index=False))

    # 8) Save to CSV if desired
    out_csv = os.getenv('OUTPUT_CSV', 'candidates.csv')
    final_df.to_csv(out_csv, index=False)
    print(f"\nSaved full candidate list to {out_csv}")


if __name__ == '__main__':
    main()
