"""
text_preprocess.py
──────────────────
Utilities for
1) normalising raw input text
2) sliding-window chunking (token- or sentence-level)

▸ clean_text(paragraph)              -> str  
▸ chunk_tokens(text, tokenizer, win, stride) -> List[str]  
▸ chunk_sentences(text, n_sent, stride)      -> List[str]
"""
from __future__ import annotations
import re, itertools
from typing import List, Iterator
import nltk
from transformers import AutoTokenizer

# Make sure the Punkt model is available the first time
try:
    _ = nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

nltk.download('punkt_tab') 

# ─────────────────────────────────────────────
# 1. Basic clean-up
# ─────────────────────────────────────────────
_WS_RE = re.compile(r"\s+")          # collapse whitespace

def clean_text(paragraph: str) -> str:
    """
    Strip leading/trailing spaces, collapse runs of whitespace,
    and normalise quotes.
    """
    txt = paragraph.strip()
    txt = _WS_RE.sub(" ", txt)
    # Normalise curly quotes / dashes, if needed
    txt = txt.replace("“", "\"").replace("”", "\"") \
             .replace("’", "'").replace("–", "-")
    return txt


# ─────────────────────────────────────────────
# 2. Token-level sliding window
# ─────────────────────────────────────────────
def _sliding_windows(seq: List[int], win: int, stride: int) -> Iterator[List[int]]:
    """
    Generic helper: yield overlapping slices of `seq`
    of length `win`, stepping `stride`.
    """
    if win <= 0 or stride <= 0:
        raise ValueError("`win` and `stride` must be positive")
    for start in range(0, len(seq), stride):
        end = start + win
        window = seq[start:end]
        if len(window) < win // 2:          # stop when last slice is too small
            break
        yield window

def chunk_tokens(text: str,
                 tokenizer: AutoTokenizer,
                 win: int = 128,
                 stride: int = 64) -> List[str]:
    """
    Split `text` into overlapping token windows (BPE IDs → back to string).
    The default window/stride (128 / 64) works well for LLaMA-3 8B context,
    but tweak as you like.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    decoded_chunks = [
        tokenizer.decode(chunk,
                         skip_special_tokens=True,
                         clean_up_tokenization_spaces=True).strip()
        for chunk in _sliding_windows(ids, win, stride)
    ]
    return decoded_chunks


# ─────────────────────────────────────────────
# 3. Sentence-level sliding window
# ─────────────────────────────────────────────
def chunk_sentences(text: str,
                    n_sent: int = 3,
                    stride: int = 1) -> List[str]:
    """
    Overlapping sentence windows.  n_sent = how many sentences per chunk;
    stride = how many sentences to move each step.
    """
    sents = nltk.sent_tokenize(text)
    if not sents:
        return []
    windows = (
        " ".join(sents[i : i + n_sent]).strip()
        for i in range(0, len(sents), stride)
        if len(sents[i : i + n_sent]) >= max(1, n_sent // 2)
    )
    return list(windows)


# ─────────────────────────────────────────────
# 4. Quick demo
# ─────────────────────────────────────────────
if __name__ == "__main__":
    demo_para = """
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

    # 0) clean
    clean = clean_text(demo_para)
    print("Cleaned text:\n", clean, "\n")

    # 1) token-window chunks
    tok    = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    tok_chunks = chunk_tokens(clean, tok, win=64, stride=32)
    print("Token windows:")
    for i, c in enumerate(tok_chunks, 1):
        print(f"[{i}] {c}\n")

    # 2) sentence-window chunks
    sent_chunks = chunk_sentences(clean, n_sent=2, stride=1)
    print("Sentence windows:")
    for i, c in enumerate(sent_chunks, 1):
        print(f"[{i}] {c}\n")
