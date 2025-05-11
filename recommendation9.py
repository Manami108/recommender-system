

# - Paper embedding yes
# - Knowledge graph no
# - Based on topic 
# - Knowledge graph embedding no
# - Reasoning no
# - Diversity no
# - Similarity yes

import torch
import numpy as np
import pandas as pd
import hnswlib
from transformers import AutoTokenizer, AutoModel

# ===================================
# Paths (Update if needed)
# ===================================
embedding_file = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/embeddings_sciBERT.npy"
index_file = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/hnsw_index.bin"
df_file = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/preprocessed_dblp_FIXED.csv"

# ===================================
# Load Metadata and HNSW Index
# ===================================
print("Loading paper metadata and HNSW index...")
df = pd.read_csv(df_file)
dim = np.load(embedding_file).shape[1]

index = hnswlib.Index(space='cosine', dim=dim)
index.load_index(index_file)

# ===================================
# Load SciBERT for Paragraph Embedding
# ===================================
print("Loading SciBERT...")
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to("cuda")
model.eval()

def embed_paragraph(text):
    if not isinstance(text, str) or not text.strip():
        return np.zeros(768 * 2, dtype=np.float32)
    with torch.no_grad():
        tokens = tokenizer(text, padding="max_length", max_length=512,
                           truncation=True, return_tensors="pt").to("cuda")
        output = model(**tokens)
        cls = output.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        return np.concatenate([cls, cls])  # 1536-dim to match paper embeddings

# ===================================
# Input from Terminal
# ===================================
paragraph = input("\nEnter a paragraph you're writing:\n\n").strip()
print("\nGenerating SciBERT embedding...")
query_vector = embed_paragraph(paragraph)
query_vector /= np.linalg.norm(query_vector) + 1e-12  # Normalize

# ===================================
# Search for Top-K Papers
# ===================================
print("\nSearching for similar papers...\n")
top_k = 5
labels, distances = index.knn_query(query_vector, k=top_k)

# ===================================
# Display Results
# ===================================
print("Top 5 Recommended Papers:\n")
for rank, (idx, dist) in enumerate(zip(labels[0], distances[0])):
    paper = df.iloc[idx]
    print(f"Rank {rank + 1}")
    print(f"Paper ID : {paper['id']}")
    print(f"Title    : {paper['title']}")
    print(f"SimScore : {1.0 - dist:.4f}")
    print(f"Abstract : {paper['indexed_abstract'][:300]}...\n")
