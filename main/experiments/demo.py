# demo.py

from chain_of_thought import extract_context
from cypher_query    import retrieve_candidates

if __name__ == "__main__":
    paragraph = """
    Recently, as advanced natural language processing techniques, Large Language
    Models (LLMs) with billion parameters have generated large impacts on various
    research fields such as Natural Language Processing (NLP), Computer Vision,
    and Molecule Discovery. Technically most existing LLMs are transformer-based
    models pre-trained on a vast amount of textual data from diverse sources,
    such as articles, books, websites, and other publicly available written
    materials. As the parameter size of LLMs continues to scale up with a larger
    training corpus, recent studies indicated that LLMs can lead to the emergence
    of remarkable capabilities. More specifically, LLMs have demonstrated the
    unprecedentedly powerful abilities of their fundamental responsibilities in
    language understanding and generation. These improvements enable LLMs to
    better comprehend human intentions and generate language responses that are
    more human-like in nature. Moreover, recent studies indicated that LLMs
    exhibit impressive generalization and reasoning capabilities, making LLMs
    better generalize to a variety of unseen tasks and domains.
    """

    # ─── Stage 1: Context Extraction ───────────────────────────────────────
    ctx = extract_context(paragraph)
    print("\n— Extracted context signals —")
    for k, v in ctx.items():
        print(f"{k:<18}: {v}")

    # ─── Stage 2: Retrieve via Cypher Queries ─────────────────────────────
    candidates_df = retrieve_candidates(ctx, top_n_each=40)

    print(f"\nFound {len(candidates_df)} unique candidate papers:\n")
    print(candidates_df.head(10).to_string(index=False))
