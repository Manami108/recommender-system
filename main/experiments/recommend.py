# ------------------------ recommend.py ----------------------
"""Orchestration CLI:  python recommend.py <file_with_paragraph.txt>"""
import sys, pandas as pd
from intent_analysis import analyse
from recall import recall
from citation import expand
from rerank import rerank

# Example paragraph for demo
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

def recommend(paragraph: str):
    intent = analyse(paragraph)
    df1    = recall(intent)
    df2    = expand(df1)
    final  = rerank(paragraph, intent, df2)
    return final

if __name__ == "__main__":
    # If a file is provided, read paragraph from file, else use default sample
    if len(sys.argv) > 1:
        para = open(sys.argv[1]).read()
    else:
        para = paragraph
    res = recommend(para)
    pd.set_option("display.max_colwidth", 120)
    print(res[["pid", "title", "llm_score"]].to_string(index=False))
