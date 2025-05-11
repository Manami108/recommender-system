# - Paper embedding yes
# - Knowledge graph yes
# - Based on topic 
# - Knowledge graph embedding no
# - Reasoning no
# - Diversity no
# - Similarity yes

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import ast
from sklearn.metrics.pairwise import cosine_similarity

# Paths
node_file = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/graph_nodes.csv"
edge_file = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/graph_edges.csv"

# Load graph data
print("Loading nodes and edges...")
nodes_df = pd.read_csv(node_file)
edges_df = pd.read_csv(edge_file)

# Split node types
paper_nodes = nodes_df[nodes_df[":LABEL"] == "Paper"].copy()
topic_nodes = nodes_df[nodes_df[":LABEL"] == "Topic"].copy()
fos_nodes = nodes_df[nodes_df[":LABEL"] == "FOS"].copy()

# Load SciBERT
print("Loading SciBERT...")
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to("cuda").eval()

def embed_text(text):
    if not isinstance(text, str) or not text.strip():
        return np.zeros(768, dtype=np.float32)
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        outputs = model(**inputs)
        cls = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
    return cls.astype(np.float32)

# Get user input
paragraph = input("\nEnter a paragraph you're writing:\n\n").strip()
print("Embedding paragraph with SciBERT...")
paragraph_emb = embed_text(paragraph).reshape(1, -1)

# Calculate similarity with FOS and Topic embeddings
print("Calculating similarities...")
fos_embs = fos_nodes[[col for col in fos_nodes.columns if col.startswith("fos_emb_")]].values
topic_embs = topic_nodes[[col for col in topic_nodes.columns if col.startswith("topic_emb_")]].values

fos_sim = cosine_similarity(paragraph_emb, fos_embs)[0]
topic_sim = cosine_similarity(paragraph_emb, topic_embs)[0]

# Find top-N FOS and Topic nodes
top_k = 3
top_fos_ids = fos_nodes.iloc[np.argsort(fos_sim)[-top_k:]]["id:ID"].tolist()
top_topic_ids = topic_nodes.iloc[np.argsort(topic_sim)[-top_k:]]["id:ID"].tolist()

# Extract papers connected to top topics or FOS
print("Finding relevant papers through KG reasoning...")
candidate_papers = set()

# Add papers connected to top topics
topic_edges = edges_df[(edges_df[":TYPE"] == "HAS_TOPIC") & (edges_df[":END_ID"].astype(str).isin(top_topic_ids))]
candidate_papers.update(topic_edges[":START_ID"].tolist())

# Add papers connected to top FOS
fos_edges = edges_df[(edges_df[":TYPE"] == "HAS_FOS") & (edges_df[":END_ID"].isin(top_fos_ids))]
candidate_papers.update(fos_edges[":START_ID"].tolist())

# Filter valid papers
candidate_df = paper_nodes[paper_nodes["id:ID"].isin(candidate_papers)].copy()

# Embed titles + abstracts
print(f"Embedding {len(candidate_df)} candidate papers...")
titles = candidate_df["title"].fillna("").tolist()
abstracts = candidate_df["year"].fillna("").astype(str).tolist()
combined = [t + " " + a for t, a in zip(titles, abstracts)]
paper_embeddings = np.array([embed_text(text) for text in combined])
paper_sim = cosine_similarity(paragraph_emb, paper_embeddings)[0]

# Rank and show results
top_indices = np.argsort(paper_sim)[-5:][::-1]
top_papers = candidate_df.iloc[top_indices].copy()
top_papers["similarity"] = paper_sim[top_indices]

import ace_tools as tools; tools.display_dataframe_to_user(name="Top Recommended Papers (KG-Based)", dataframe=top_papers[["id:ID", "title", "year", "n_citation", "similarity"]])
