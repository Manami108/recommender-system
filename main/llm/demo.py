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

def main():
    # 1) Example input paragraph
    paragraph = """
This paper accordingly proposes a novel Context-guided Triple Matching (CTM) framework, which aims to address the limitations of traditional pairwise matching strategies commonly employed in semantic understanding tasks. Traditional models often rely on pairwise interactions—such as between question and passage, or passage and answer—yet fail to fully exploit the rich semantic interplay that arises from considering all three components simultaneously. In contrast, CTM introduces a principled method for incorporating the third component as a dynamic context, thereby enriching the semantic alignment between the remaining two entities.
At the core of the CTM framework lies a hierarchical attention flow mechanism designed to model and enhance the mutual information among the three components of a semantic triple, typically composed of {question, passage, answer}. The key innovation is the use of contextual priors—by sequentially treating one component as the contextual backbone and aligning the remaining two components with it through attentive operations.
Concretely, given a candidate triple, the framework begins by selecting one of the three elements as the context provider, denoted as the prior context. This prior is then used to compute bidirectional attention weights with each of the other two components individually. This dual attention mechanism not only captures token-level alignment but also serves to emphasize the contextual saliency of semantic cues that may otherwise be overlooked in isolated pairwise comparisons.
Once the two sets of attentional correlations are computed (context-to-component A and context-to-component B), a secondary fusion layer is introduced. This layer aggregates the two attention maps via a gated fusion strategy or a weighted average, depending on implementation. The result is a context-aware representation that jointly encapsulates semantic dependencies between the context and the other two components. This enables the model to more holistically model relationships, such as how an answer relates to both the question and supporting passage when either of them serves as the context.
To ensure symmetry and completeness, this process is repeated across all three permutations of the triple, where each component alternately plays the role of the prior context. This results in three distinct context-aware representations, each capturing a unique perspective of the semantic triangle. These representations are then concatenated and passed through a shared fully connected projection layer, which estimates the final matching score for the triple. This scoring function can be interpreted as the model’s confidence in the semantic coherence of the triple under the CTM encoding.
To further enhance the discriminative power of the model, especially in fine-grained reasoning tasks where subtle semantic differences must be discerned, we introduce a contrastive regularization objective. Specifically, during training, the model is encouraged to maximize the cosine similarity between context-aware representations of ground-truth triples, while simultaneously pushing apart representations of distractor or mismatched triples in the latent space. This contrastive learning component acts as a semantic discriminator, enabling the model to differentiate between closely competing answer candidates based on nuanced contextual clues.
Overall, the proposed CTM framework offers several key advantages. First, by leveraging all components in a rotational and hierarchical manner, it avoids the asymmetry and incompleteness inherent in pairwise-only models. Second, the hierarchical attention flow promotes deeper interaction and information flow between components, which is particularly beneficial in scenarios involving implicit reasoning or multi-hop inference. Third, the integration of contrastive regularization introduces robustness against semantically similar but incorrect candidates—thereby improving both precision and generalization.
Extensive empirical evaluations on benchmark datasets across reading comprehension, multiple-choice QA, and passage ranking demonstrate that the CTM framework consistently outperforms strong baselines, particularly in tasks requiring complex relational understanding. Ablation studies further reveal the contribution of each component, confirming that context rotation and contrastive loss both significantly enhance model performance.
In summary, CTM represents a significant step forward in semantic matching by unifying hierarchical attention, contextual priors, and contrastive reasoning in a single end-to-end architecture. It provides a flexible yet powerful mechanism for modeling triadic relationships in natural language and holds promise for a wide range of downstream NLP tasks.
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
    print(topk[['pid', 'title', 'rank', 'justification']].to_string(index=False))

    # 8) Save final results
    out_csv = os.getenv('OUTPUT_CSV', 'final_recommendations.csv')
    topk.to_csv(out_csv, index=False)
    print(f"6) Saved final recommendations to {out_csv}")

if __name__ == '__main__':
    main()
