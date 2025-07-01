# This code splits the paragraph (user's query) into overlapping chunks for LLM (context window)

from __future__ import annotations
import re # text normalization
from typing import List, Iterator
import nltk # natural language toolkit used for sentence toknization
from transformers import AutoTokenizer
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize.texttiling import TextTilingTokenizer

# Installing NLTK Punkt tokenizer
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
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
                 win: int = 256,
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
# component from the triple as tzhe prior context. Then we apply the bidirectional attention to
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

# ─────────────────────────────── DEMO USAGE ─────────────────────────────────
# if __name__ == '__main__':
#     demo = """
# Humans use the gaze to look at objects. This behavior can be used as a means to control interfaces in human–computer interaction by estimating the gaze point with the help of an eye tracker. 
# Thanks to recent technological advancements and drop in price, eye tracking is no longer a niche technology only used in laboratories or by users with special needs. For example, with the price of an advanced game controller, players can enhance their gaming experience with eye tracking. 
# A gaze-aware game knows where the player’s visual attention is at each moment and can offer optional input methods and enhanced gaming experience. At the same time, research on mobile eye tracking has been active. Simple eye-awareness is already included in some cell phone models so that the phone knows when the user is looking at it. 
# Research on pervasive and mobile gaze interaction has demonstrated how eye tracking can enhance the interaction with mobile phones, tablets, smartwatches, smart glasses, and smart environments and public displays. 
# Because the eye is primarily a perceptual organ, using gaze as an intentional control method poses challenges for interaction design. Most important, viewing should not be misinterpreted as a voluntary command. In gaze interaction literature, this problem is known as the Midas touch problem, where viewed objects are unintentionally acted on. 
# Feedback plays an essential role in informing the user how the system is interpreting the gaze. Gazing at an object in real life naturally provides only visual feedback. But computers and smart devices can indicate if an object has been recognized as being pointed at, or being selected. Previous research has shown that visual and auditory feedback on gaze input significantly improve user performance and satisfaction. 
# However, the effects of haptic feedback in gaze interaction have remained largely unknown. We assume haptic feedback could provide a useful alternative to, at least the audio, as auditory and haptic perception are known to share similarities. For example, participants could perceive auditory and tactile rhythms more accurately than visual rhythms. Auditory and haptic feedback can be perceived independently from the gaze location. 
# Unlike the distal senses of vision and hearing, touch is a proximal sense that provides information of things close to or in contact with us. How would the interplay of a distal and proximal sense work? For instance, instead of seeing a button change its appearance, the user could feel the click of a button after selecting it with gaze. Could this novel combination of modalities provide some benefits compared to visual and auditory feedback, or is this unnatural combination of action and feedback perhaps incomprehensible? 
# These were the questions that motivated us in the work reported in this article. Haptic feedback has become more common in consumer technology due to the emergence of mobile and wearable devices designed to be in contact with the skin. The most popular form of haptic stimulation in mobile and wearable devices is vibrotactile feedback. For example, continuous vibration is an effective way to notify of incoming calls with mobile phones. 
# Shorter vibration bursts are used on phones and tablets to replace the tactile feel of pressing a physical key when typing with a virtual keyboard. This has been shown to improve typing speeds. Vibrotactile technology is also included in some smartwatches. In the Apple Watch, for instance, vibrotactile stimulation is used to mimic a heartbeat that can be sent to a close friend or family member. With multiple actuators, it is possible to create touch sensations that move on the wrist. 
# To date, commercial smart glasses and other head-mounted devices have not utilized vibrotactile feedback. This is surprising because it is known that users can quite accurately localize which part of the head is stimulated with vibrotactile actuators. 
# We were interested in studying how vibrotactile feedback could support gaze interaction. We conducted a series of experiments in which we focused on four main research questions: effectiveness of vibrotactile feedback, temporal limits between gaze events and vibrotactile feedback, effects of feedback location and spatial setup, and vibrotactile feedback in comparison to other modalities. Because our results are spread over more than 20 articles, this could make it difficult for future researchers to extract the main findings. 
# The contribution of this article is to summarize the research results in a compact form and serve as a collection of pointers to more detailed work in the original publications. The goal is to add to our understanding of how the two modalities of haptics and gaze can be utilized effectively in human–computer interaction. The organization of the article is as follows. We first introduce gaze interaction and vibrotactile feedback. We then present results from the experiments before discussing lessons learned from the studies. We end with general discussion and present design guidelines based on our accumulated knowledge and insights.
#     """
#     clean = clean_text(demo)
#     tok = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', use_fast=True)

#     print('--- Token Chunks ---')
#     for c in chunk_tokens(clean, tok): print(c, '\n')

#     print('--- Sentence Chunks ---')
#     for c in chunk_sentences(clean): print(c, '\n')
