#!/usr/bin/env python
from bertopic import BERTopic
from bertopic.vectorizers import OnlineCountVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sentence_transformers import SentenceTransformer

import pandas as pd, numpy as np, torch, gc, os, csv, json, ast, re

# ─── Paths & Hyperparameters ─────────────────────────────────────────────
CSV          = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/preprocessed_dblp_FIXED.csv"
OUT_CSV      = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/topic_modeled_dblp.csv"
MODEL_PATH   = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/models/bertopic_online.pkl"

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_DOCS   = 10_000
BATCH_EMB    = 256
N_CLUSTERS   = 100
VOCAB_MIN_DF = 2

# ─── Cleaning and Decoding ──────────────────────────────────────────────
_re_hex      = re.compile(r"\\x[0-9a-fA-F]{2}")
_re_unicode  = re.compile(r"\\u[0-9a-fA-F]{4}")
_re_punct    = re.compile(r"[^\w\s\-\.,;:()]")

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
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def build_docs(chunk: pd.DataFrame) -> list[str]:
    docs = []
    for title, abs_raw in zip(chunk["title"].fillna(""), chunk["indexed_abstract"].fillna("")):
        title_clean = title.rstrip("/").strip()
        abstract    = decode_indexed_abstract(abs_raw)
        docs.append(clean_text(f"{title_clean} {abstract}"))
    return docs

# ─── Models ──────────────────────────────────────────────────────────────
print("▶ Loading embedding model …")
sbert      = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
vectorizer = OnlineCountVectorizer(stop_words="english",
                                   ngram_range=(1, 2),
                                   min_df=VOCAB_MIN_DF)
clusterer  = MiniBatchKMeans(n_clusters=N_CLUSTERS,
                             batch_size=2048,
                             random_state=42)
dr_model   = IncrementalPCA(n_components=5)

topic_model = BERTopic(
    embedding_model=None,
    umap_model=dr_model,
    hdbscan_model=clusterer,
    vectorizer_model=vectorizer,
    calculate_probabilities=False,
    verbose=True
)

# ─── Online Training ─────────────────────────────────────────────────────
print("▶ Starting training …")
all_docs, all_topics = [], []
reader = pd.read_csv(CSV, usecols=["title", "indexed_abstract"], chunksize=CHUNK_DOCS)

for chunk in reader:
    docs = build_docs(chunk)
    emb  = sbert.encode(docs, batch_size=BATCH_EMB,
                        convert_to_numpy=True).astype(np.float64, copy=False)
    
    # Training and Topic Assignment
    topic_model.partial_fit(docs, embeddings=emb)
    topics, _ = topic_model.transform(docs, embeddings=emb)  # Unpack tuple (topics, probs)

    all_docs.extend(docs)
    all_topics.extend(topics)

    torch.cuda.empty_cache(); gc.collect()
    print(f"✓ Trained on {len(all_docs):,} docs")

# ─── Update Keywords for All Topics ──────────────────────────────────────
topic_model.update_topics(all_docs, topics=all_topics, vectorizer_model=vectorizer)

# ─── Save Model and Topic-Labeled CSV ────────────────────────────────────
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
topic_model.save(MODEL_PATH)
print(f"✓ Model saved to: {MODEL_PATH}")

pd.DataFrame({"doc": all_docs, "topic": all_topics}).to_csv(OUT_CSV, index=False)
print(f"✓ Labeled docs saved to: {OUT_CSV}")

# ─── Visualization (Top 25 Topics) ──────────────────────────────────────
print("▶ Creating visualizations …")

CHART_HTML = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/topics_barchart.html"

fig = topic_model.visualize_barchart(top_n_topics=25)
fig.write_html(CHART_HTML)
print(f"✓ Topic bar chart saved to: {CHART_HTML}")
