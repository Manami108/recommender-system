import pandas as pd
import numpy as np
import torch, gc, os, json, ast, re
from bertopic import BERTopic
from bertopic.vectorizers import OnlineCountVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sentence_transformers import SentenceTransformer
from nltk.stem import WordNetLemmatizer
import nltk

# ─── NLTK Setup ──────────────────────────────────────────────────────────
nltk.download("wordnet")
nltk.download("omw-1.4")
lemmatizer = WordNetLemmatizer()

# ─── Paths ───────────────────────────────────────────────────────────────
CSV_ORIGINAL = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/preprocessed_dblp_FIXED.csv"
CSV_OUTPUT   = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/topic_modeled_dblp.csv"
MODEL_PATH   = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/models/bertopic_online.pkl"
CHART_HTML   = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/topics_barchart.html"

# ─── Parameters ──────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_DOCS   = 10_000
BATCH_EMB    = 256
N_CLUSTERS   = 200
VOCAB_MIN_DF = 2

# ─── Compile Regex ───────────────────────────────────────────────────────
_re_hex     = re.compile(r"\\x[0-9a-fA-F]{2}")
_re_unicode = re.compile(r"\\u[0-9a-fA-F]{4}")
_re_punct   = re.compile(r"[^\w\s\-\.,;:()]")

# ─── Preprocessing Functions ─────────────────────────────────────────────
def decode_indexed_abstract(raw: str) -> str:
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
        inv = obj.get("InvertedIndex", {})
        length = obj.get("IndexLength", max((max(pos) for pos in inv.values()), default=-1) + 1)
        tokens = [""] * length
        for word, positions in inv.items():
            for p in positions:
                if 0 <= p < length:
                    tokens[p] = word
        return " ".join(t for t in tokens if t)
    return s

def clean_text(text: str) -> str:
    text = _re_hex.sub(" ", text)
    text = _re_unicode.sub(" ", text)
    text = _re_punct.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join([lemmatizer.lemmatize(word.lower()) for word in text.split()])

def build_docs(chunk: pd.DataFrame) -> list[str]:
    docs = []
    for title, abs_raw in zip(chunk["title"].fillna(""), chunk["indexed_abstract"].fillna("")):
        title_clean = title.rstrip("/").strip()
        abstract = decode_indexed_abstract(abs_raw)
        docs.append(clean_text(f"{title_clean} {abstract}"))
    return docs

# ─── Load Model Components ───────────────────────────────────────────────
print("▶ Loading embedding model …")
sbert = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
vectorizer = OnlineCountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=VOCAB_MIN_DF)
clusterer = MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=2048, random_state=42)
dr_model = IncrementalPCA(n_components=5)

topic_model = BERTopic(
    embedding_model=None,
    umap_model=dr_model,
    hdbscan_model=clusterer,
    vectorizer_model=vectorizer,
    calculate_probabilities=False,
    verbose=True
)

# ─── Run BERTopic and Collect Topics ─────────────────────────────────────
print("▶ Starting training …")
all_docs, all_topics = [], []
doc_topic_list = []

reader = pd.read_csv(CSV_ORIGINAL, usecols=["title", "indexed_abstract"], chunksize=CHUNK_DOCS)

for chunk in reader:
    docs = build_docs(chunk)
    emb  = sbert.encode(docs, batch_size=BATCH_EMB, convert_to_numpy=True).astype(np.float64, copy=False)

    topic_model.partial_fit(docs, embeddings=emb)
    topics, _ = topic_model.transform(docs, embeddings=emb)

    all_docs.extend(docs)
    all_topics.extend(topics)
    doc_topic_list.extend(topics)

    torch.cuda.empty_cache(); gc.collect()
    print(f"✓ Trained on {len(all_docs):,} docs")

# ─── Update Topics Globally ──────────────────────────────────────────────
topic_model.update_topics(all_docs, topics=all_topics, vectorizer_model=vectorizer)

# ─── Save Trained Model and Chart ────────────────────────────────────────
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
topic_model.save(MODEL_PATH)
print(f"✓ Model saved to: {MODEL_PATH}")

fig = topic_model.visualize_barchart(top_n_topics=25)
fig.write_html(CHART_HTML)
print(f"✓ Topic bar chart saved to: {CHART_HTML}")

# ─── Save Full CSV with Topics ───────────────────────────────────────────
print("▶ Saving topic-annotated CSV …")
original_df = pd.read_csv(CSV_ORIGINAL)
original_df_copy = original_df.copy()

# Safety check to avoid index misalignment
if len(original_df_copy) != len(doc_topic_list):
    raise ValueError("Mismatch: number of topic labels does not match number of original documents.")

original_df_copy["topic"] = doc_topic_list
original_df_copy.to_csv(CSV_OUTPUT, index=False)
print(f"✓ Full dataset with topics saved to: {CSV_OUTPUT}")
