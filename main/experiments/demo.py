import os
import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from chunking import clean_text, chunk_tokens
from recall import (
    recall_by_chunks,
    recall_fulltext,
    recall_vector,
    embed,
    fetch_metadata,
)
from hop_reasoning import multi_hop_topic_citation_reasoning
from rerank_llm import llm_contextual_rerank

# ─────────────────────────────────────────────────────────────────────────────
# Demo script: full pipeline with LLaMA contextual coherence reranking
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Example input paragraph
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

    # 2) Clean & chunk the paragraph (token-level)
    cleaned   = clean_text(paragraph)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    chunks    = chunk_tokens(cleaned, tokenizer, win=128, stride=64)
    print(f"1) Generated {len(chunks)} chunks")

    # 3a) Global/full-paragraph recall
    full_bm25 = recall_fulltext(cleaned, k=40).assign(source='bm25_full')
    full_vec  = recall_vector(embed(cleaned), k=40, sim_threshold=0.30).assign(source='embed_full')
    print(f"2a) Full-paragraph BM25 hits: {len(full_bm25)}, embed hits: {len(full_vec)}")

    # 3b) Chunk-based recall
    chunk_df = recall_by_chunks(chunks, k_bm25=40, k_vec=40, sim_th=0.30).assign(source='chunked')
    print(f"2b) Chunk-based recall candidates: {len(chunk_df)}")

    # 3c) Combine & dedupe
    rec_df = pd.concat([full_bm25, full_vec, chunk_df], ignore_index=True)
    rec_df = (
        rec_df
        .sort_values(['source','sim'], ascending=[True, False])
        .drop_duplicates('pid', keep='first')
        .reset_index(drop=True)
    )
    print(f"2) Total unique recall candidates: {len(rec_df)}")

    # # 4) Multi-hop reasoning (topics/FoS + citations)
    # seed_pids = rec_df['pid'].tolist()
    # cit_df = multi_hop_topic_citation_reasoning(
    #     seed_pids,
    #     max_topic_hops=2,
    #     top_n=100
    # )
    # print(f"3) Retrieved {len(cit_df)} reasoning candidates")

    # # 5) Merge reasoning + recall & dedupe
    # combined = pd.concat([
    #     rec_df.assign(source='recall'),
    #     cit_df.assign(source='reasoning', sim=np.nan)
    # ], ignore_index=True)
    # combined = (
    #     combined
    #     .sort_values(['hop','sim'], ascending=[True, False])
    #     .drop_duplicates('pid', keep='first')
    #     .reset_index(drop=True)
    # )
    # print(f"4) Merged to {len(combined)} unique candidates")

    # 6) Fetch abstracts for reranking
    meta = fetch_metadata(rec_df['pid'].tolist())
    df   = rec_df.merge(meta[['pid','abstract']], on='pid', how='left')

    # 7) LLaMA contextual coherence rerank
    reranked = llm_contextual_rerank(paragraph, df[['pid','title','abstract']])
    topk     = reranked.head(10)
    print("5) Top 10 recommendations after contextual coherence rerank:")
    print(topk[['pid','title','final_score']].to_string(index=False))

    # 8) Save final results
    out_csv = os.getenv('OUTPUT_CSV', 'final_recommendations.csv')
    topk.to_csv(out_csv, index=False)
    print(f"6) Saved final recommendations to {out_csv}")

if __name__ == '__main__':
    main()
