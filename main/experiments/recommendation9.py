# - Paper embedding yes
# - Knowledge graph no
# - Based on topic no
# - Knowledge graph embedding no
# - Reasoning no
# - Diversity no
# - Similarity yes

import json
import numpy as np
import pandas as pd
import hnswlib
from transformers import AutoTokenizer, AutoModel
import torch
import ast
from tqdm import tqdm
from sklearn.metrics import ndcg_score

# ─── File Paths ──────────────────────────────────────────────────────────
embedding_file = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/big/embeddings_sciBERT.npy"
index_file = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/big/hnsw_index.bin"
df_file = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/big/dblp.v12.csv"
test_jsonl = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/big/testset_150.jsonl"

# ─── Load Paper Metadata and Embedding Index ─────────────────────────────
df = pd.read_csv(df_file, low_memory=False, dtype=str)
paper_id_map = {str(k): i for i, k in enumerate(df["id"].astype(str))}

index = hnswlib.Index(space='cosine', dim=np.load(embedding_file).shape[1])
index.load_index(index_file)

# ─── Load SciBERT ────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to("cuda")
model.eval()

# ─── Helper Functions ────────────────────────────────────────────────────
def embed_paragraph(text):
    if not isinstance(text, str) or not text.strip():
        return np.zeros(1536, dtype=np.float32)
    with torch.no_grad():
        tokens = tokenizer(text, padding="max_length", max_length=512,
                           truncation=True, return_tensors="pt").to("cuda")
        output = model(**tokens)
        cls = output.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        return np.concatenate([cls, cls])

def compute_metrics(top_ids, true_ids):
    true_set = set(true_ids)
    hit = [1 if pid in true_set else 0 for pid in top_ids]
    precision = sum(hit) / len(top_ids)
    recall = sum(hit) / len(true_ids) if true_ids else 0
    hr = 1.0 if any(hit) else 0
    relevance = [1 if pid in true_set else 0 for pid in top_ids]
    ideal = sorted(relevance, reverse=True)
    ndcg = ndcg_score([ideal], [relevance]) if any(relevance) else 0
    return precision, recall, hr, ndcg

# ─── Evaluate on Test Set ────────────────────────────────────────────────
results = []
with open(test_jsonl, "r", encoding="utf-8") as f:
    for line in tqdm(f, total=150):
        obj = json.loads(line)
        para = obj["paragraph"]
        references = [str(r) for r in obj.get("references", []) if str(r) in paper_id_map]

        if not references:
            continue

        query_vec = embed_paragraph(para)
        query_vec /= np.linalg.norm(query_vec) + 1e-12

        labels, _ = index.knn_query(query_vec, k=5)
        top_pred_ids = [str(df.iloc[i]["id"]) for i in labels[0]]

        precision, recall, hr, ndcg = compute_metrics(top_pred_ids, references)
        results.append((precision, recall, hr, ndcg))


# ─── Aggregate Results ───────────────────────────────────────────────────
results = np.array(results)
avg_precision, avg_recall, avg_hr, avg_ndcg = results.mean(axis=0)

metrics = {
    "Precision@5": round(avg_precision, 4),
    "Recall@5": round(avg_recall, 4),
    "HR@5": round(avg_hr, 4),
    "NDCG@5": round(avg_ndcg, 4),
    "Evaluated Samples": len(results)
}

print("\nEvaluation Results:")
for key, value in metrics.items():
    print(f"{key}: {value}")
