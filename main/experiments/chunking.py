# This code splits the paragraph (user's query) into overlapping chunks for LLM (context window)

from __future__ import annotations
import re, itertools # text normalization
from typing import List, Iterator
import nltk # natural language toolkit used for sentence toknization
from transformers import AutoTokenizer

# Installing NLTK Punkt tokenizer
try:
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


# user query text cleaning and processing 
# _WS_RE ensures no multiple spaces and double lines and so on
_WS_RE = re.compile(r"\s+")          

def clean_text(paragraph: str) -> str:
    txt = paragraph.strip()
    txt = _WS_RE.sub(" ", txt)
    txt = txt.replace("“", "\"").replace("”", "\"") \
             .replace("’", "'").replace("–", "-")
    return txt

# sliding windows
# stride is how far to move the window each time 
def _sliding_windows(seq: List[int], win: int, stride: int) -> Iterator[List[int]]:
    if win <= 0 or stride <= 0:
        raise ValueError("`win` and `stride` must be positive")
    for start in range(0, len(seq), stride):
        end = start + win
        window = seq[start:end]
        if len(window) < win // 2:          # stop when last slice is too small
            break
        yield window

# token level chunking 
# encode full text to token ID -> break ids into overlapping windows via sliding windows -> decode each id window back int a string 
# now window is set to 128, stride is set to 64, but needs reference and experiment 
def chunk_tokens(text: str,
                 tokenizer: AutoTokenizer,
                 win: int = 128,
                 stride: int = 64) -> List[str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    decoded_chunks = [
        tokenizer.decode(chunk,
                         skip_special_tokens=True,
                         clean_up_tokenization_spaces=True).strip()
        for chunk in _sliding_windows(ids, win, stride)
    ]
    return decoded_chunks

# Sentence level chunking 
# number of sentence per a chunk is set to 3, and stride is set to 1 but need reference and experiment 
# This is more semantic window aligned to natural sentence boundries. 
def chunk_sentences(text: str,
                    n_sent: int = 3,
                    stride: int = 1) -> List[str]:
    sents = nltk.sent_tokenize(text)
    if not sents:
        return []
    windows = (
        " ".join(sents[i : i + n_sent]).strip()
        for i in range(0, len(sents), stride)
        if len(sents[i : i + n_sent]) >= max(1, n_sent // 2)
    )
    return list(windows)


# # ─────────────────────────────────────────────
# # 4. Quick demo
# # ─────────────────────────────────────────────
# if __name__ == "__main__":
#     demo_para = """
# This paper accordingly proposes a novel Context-guided Triple Matching (CTM),
# while the third component missing from the pairwise matching is adopted as a prior context.
# The proposed triple matching is present as a hierarchical attention flow to adequately capture
# the semantic relationship. Specifically, given a candidate triple, we first employ (any) one
# component from the triple as the prior context. Then we apply the bidirectional attention to
# calculate the correlation between context and the other two components separately. Afterwards,
# another attention layer is utilized to leverage two above correlations to form an aggregated
# context-aware representation. In this way, the model is able to gather more comprehensive
# semantic relationship for the triple, according to the selected context. Similarly, we enumerate
# the other two components (from the triple) and cast as the prior context to repeat the same
# attention flow. Finally, a fully-connected layer is employed for all formed context-aware
# representations to estimate the matching score. In addition to the triple matching, we also
# consider to adopt a contrastive regularization in capturing the subtle semantic differences among
# answer candidates. The aim is to maximize the similarity of features from correct triple(s) while
# pushing away that of distractive ones, that has been neglected by existing methods.
#     """

#     # 0) clean
#     clean = clean_text(demo_para)
#     print("Cleaned text:\n", clean, "\n")

#     # 1) token-window chunks
#     tok    = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
#     tok_chunks = chunk_tokens(clean, tok, win=64, stride=32)
#     print("Token windows:")
#     for i, c in enumerate(tok_chunks, 1):
#         print(f"[{i}] {c}\n")

#     # 2) sentence-window chunks
#     sent_chunks = chunk_sentences(clean, n_sent=2, stride=1)
#     print("Sentence windows:")
#     for i, c in enumerate(sent_chunks, 1):
#         print(f"[{i}] {c}\n")
