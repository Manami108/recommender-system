
#!/usr/bin/env python

# 1) Stream papers in CHUNK_DOCS batches → online MiniBatch‑KMeans
# 2) Write topics straight back to the dataframe (no huge Python lists)
# 3) After streaming, build topic labels & vectors once (sampled or full)
# 4) Visualise and save artefacts
# Tested on: 24 GB RTX 4070, 124 GB RAM, Python 3.10, BERTopic 0.17
# careful with batch size

import os, gc, re, json, ast, warnings, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch, nltk
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

from bertopic import BERTopic
from bertopic.vectorizers import OnlineCountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction import text

# Reduce OpenMP thread explosion that sometimes seg‑faults in KMeans
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import bertopic._bertopic as _bt
_orig_c_tf_idf           = _bt.BERTopic._c_tf_idf
_orig_extract_words      = _bt.BERTopic._extract_words_per_topic
_orig_create_topic_vecs  = _bt.BERTopic._create_topic_vectors
_bt.BERTopic._c_tf_idf                = lambda self, docs, partial_fit=False: (None, [])
_bt.BERTopic._extract_words_per_topic  = lambda *_, **__: {}
_bt.BERTopic._create_topic_vectors     = lambda *_: None

ROOT         = Path("/home/abhi/Desktop/Manami")
CSV_ORIGINAL = ROOT / "dblp.v12.csv"
CSV_OUTPUT   = ROOT / "topic_modeled_SciBERT_tuned2.csv"
MODEL_PATH   = ROOT / "bertopic_online_SciBERT_tuned2.pkl"
CHART_HTML   = ROOT / "topics_barchart_SciBERT_tuned2.html"

# hyperparameters
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_DOCS      = 8_000          
BATCH_EMB       = 32             # fits in 24 GB
N_CLUSTERS      = 150
VOCAB_MIN_DF    = 15
VOCAB_MAX_FEA   = 100_000
FINAL_SAMPLE    = 50_000         # docs for label building (set None for all)
SEED            = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# abstract reconstracting
_re_hex     = re.compile(r"\\x[0-9a-fA-F]{2}")
_re_unicode = re.compile(r"\\u[0-9a-fA-F]{4}")
_re_punct   = re.compile(r"[^\w\s\-\.,;:()]")
lemmatizer  = WordNetLemmatizer()


def decode_indexed_abstract(raw: str) -> str:
    """Rebuild DBLP's inverted‑index abstract string → plain text."""
    if not raw or pd.isna(raw):
        return ""
    s = str(raw).strip()
    if s.startswith("{") and "InvertedIndex" in s:
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            try:
                obj = ast.literal_eval(s)
            except Exception:
                return s
        inv, length = obj["InvertedIndex"], obj["IndexLength"]
        tokens = [""] * length
        for w, poses in inv.items():
            for pos in poses:
                tokens[pos] = w
        return " ".join(tokens)
    return s


def clean_text(t: str) -> str:
    t = _re_hex.sub(" ", t)
    t = _re_unicode.sub(" ", t)
    t = _re_punct.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return " ".join(lemmatizer.lemmatize(w.lower()) for w in t.split())


def build_docs(chunk: pd.DataFrame) -> list[str]:
    docs = []
    for ttl, abs_raw in zip(chunk["title"].fillna(""),
                            chunk["indexed_abstract"].fillna("")):
        raw = f"{ttl.rstrip('/').strip()} {decode_indexed_abstract(abs_raw)}"
        docs.append(clean_text(raw))
    return docs


# encoeder
print("Loading SciBERT …")
encoder = SentenceTransformer("allenai/scibert_scivocab_uncased", device=DEVICE)
encoder.max_seq_length = 256      

def embed(texts: list[str]) -> np.ndarray:
    with torch.inference_mode():
        vec = encoder.encode(
            texts,
            batch_size=BATCH_EMB,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32", copy=False)   # float32 is enough for KMeans
    torch.cuda.empty_cache()
    return vec


# stop words
SCI_STOP = {
    "paper","study","result","method","approach","based","using","data","model","analysis",
    "research","proposed","algorithm","system","new","show","time","work","task","problem",
    "performance","evaluation","present","provide","find","demonstrate","significant","application",
    "develop","experiment","solution","technique","feature","important","information","achieve",
    "objective","contribute","improve","evaluate","examine","investigate","address","consider",
    "discuss","describe","summarize","review","survey","et","al","state","art","first","also", "used",
    "include","provide","set","result","find","show","use","based","propose","demonstrate",
    "achieve","present","develop","system","method","approach","algorithm","model","data", "example",
    "world", "wide", "special", "issue", "different", "kind", "international", "conference", "journal",
    "research", "paper", "proceeding", "workshop", "work", "study", "result", "method", "approach",
}
STOP_WORDS = text.ENGLISH_STOP_WORDS.union(SCI_STOP)

vectorizer = OnlineCountVectorizer(
    stop_words=STOP_WORDS,
    token_pattern=r"(?u)\b[A-Za-z][A-Za-z]+\b",
    ngram_range=(2, 3),
    min_df=VOCAB_MIN_DF,
    max_features=VOCAB_MAX_FEA,
)

clusterer = MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=2048, random_state=SEED)

topic_model = BERTopic(
    embedding_model=encoder,
    umap_model=None,            # we add UMAP only for visualisation later
    hdbscan_model=clusterer,
    vectorizer_model=vectorizer,
    representation_model=None,
    calculate_probabilities=False,
    verbose=True,
)

print("Loading CSV …")
df = pd.read_csv(CSV_ORIGINAL, low_memory=False)
assert {"title", "indexed_abstract"}.issubset(df.columns)

# Prepare empty columns
if "clean_text" not in df.columns:
    df["clean_text"] = ""
if "topic" not in df.columns:
    df["topic"] = -1

print("Online clustering …")
for start in range(0, len(df), CHUNK_DOCS):
    chunk      = df.iloc[start:start + CHUNK_DOCS]
    docs       = build_docs(chunk)
    emb        = embed(docs)

    topic_model.partial_fit(docs, embeddings=emb)
    topics, _  = topic_model.transform(docs, embeddings=emb)

    # write results back into the main dataframe, then free memory
    df.loc[chunk.index, "clean_text"] = docs
    df.loc[chunk.index, "topic"]      = topics

    print(f"✓ processed {start + len(docs):,} docs")
    del docs, emb, topics, chunk
    gc.collect()

# restore original bertopicc method
_bt.BERTopic._c_tf_idf               = _orig_c_tf_idf
_bt.BERTopic._extract_words_per_topic = _orig_extract_words
_bt.BERTopic._create_topic_vectors    = _orig_create_topic_vecs

# labels and vectors
print("Building topic representations …")

sample_mask = (df.index if FINAL_SAMPLE is None or len(df) <= FINAL_SAMPLE
               else np.random.choice(df.index, FINAL_SAMPLE, replace=False))

reps_docs   = df.loc[sample_mask, "clean_text"].tolist()
reps_topics = df.loc[sample_mask, "topic"].tolist()

# Add representation models now
topic_model.representation_model = [
    KeyBERTInspired(top_n_words=30),
    MaximalMarginalRelevance(diversity=0.95),
]

topic_model.update_topics(
    reps_docs,
    topics=reps_topics,
    vectorizer_model=vectorizer,
)

print("Post‑processing topic words …")
for tid, words_scores in topic_model.get_topics().items():
    deduped = []
    for w, s in words_scores:
        if not any(w != o and w in o for o, _ in words_scores if len(w) >= 3):
            deduped.append((w, s))
    topic_model.get_topics()[tid] = deduped

# html
print("▶ Visualising …")
try:
    from cuml.manifold import UMAP 
    topic_model.umap_model = UMAP(n_components=2, n_neighbors=10, min_dist=0.1, metric="cosine")
except ImportError:
    from umap import UMAP
    topic_model.umap_model = UMAP(n_components=2, n_neighbors=10, min_dist=0.1, metric="cosine")

fig = topic_model.visualize_barchart(top_n_topics=N_CLUSTERS)
fig.write_html(CHART_HTML)
print(f"✓ Chart  → {CHART_HTML}")

# save
print("▶ Saving model & CSV …")
(topic_model.save(MODEL_PATH), df.to_csv(CSV_OUTPUT, index=False))
print(f"✓ Model  → {MODEL_PATH}")
print(f"✓ CSV    → {CSV_OUTPUT}")

