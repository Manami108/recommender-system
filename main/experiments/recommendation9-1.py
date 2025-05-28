# - Paper embedding yes
# - Knowledge graph yes
# - Based on topic no? but the topic is included in knowledge graph tho
# - Knowledge graph embedding yes
# - Reasoning no
# - Diversity no
# - Similarity yes

import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import ast
from tqdm import tqdm
from sklearn.metrics import ndcg_score

# ─── File Paths ──────────────────────────────────────────────────────────
transE_emb_path = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/small_kg_transE_entity_embeddings.npy"
transE_id_path = transE_emb_path.replace(".npy", "_ids.txt")
csv_metadata_path = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/small_topic_modeled_SciBERT.csv"
sciBERT_emb_path = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/small_embeddings_sciBERT.npy"
projector_path = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/models/scibert_to_transE_projector.pt"
test_jsonl = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/big/testset_150.jsonl"

# ─── Load Data ───────────────────────────────────────────────────────────
transE_embeddings = np.load(transE_emb_path)
with open(transE_id_path, "r") as f:
    transE_ids = [line.strip() for line in f]

metadata_df = pd.read_csv(csv_metadata_path, low_memory=False, dtype=str)
metadata_df["title"] = metadata_df["title"].fillna("")
metadata_df["indexed_abstract"] = metadata_df["indexed_abstract"].fillna("")
paper_id_map = {str(row["id"]): idx for idx, row in metadata_df.iterrows()}

# ─── Reconstruct Abstract ────────────────────────────────────────────────
def reconstruct_abstract(indexed_str):
    try:
        obj = ast.literal_eval(indexed_str)
        idx_len = obj.get("IndexLength", 0)
        inv_index = obj.get("InvertedIndex", {})
        tokens = [""] * idx_len
        for word, pos_list in inv_index.items():
            for pos in pos_list:
                if 0 <= pos < idx_len:
                    tokens[pos] = word
        return " ".join(tokens).strip()
    except:
        return ""

# ─── Filter Paper Embeddings ─────────────────────────────────────────────
paper_embeddings, paper_records, paper_sciBERT = [], [], []
sciBERT_all = np.load(sciBERT_emb_path)

for eid, emb in zip(transE_ids, transE_embeddings):
    if eid.startswith("paper_"):
        idx = int(eid.split("_")[1])
        if idx < len(metadata_df):
            row = metadata_df.iloc[idx]
            real_id = row["id"]
            paper_embeddings.append(emb)
            paper_sciBERT.append(sciBERT_all[idx])
            paper_records.append({
                "id": real_id,
                "title": row["title"],
                "abstract": reconstruct_abstract(row["indexed_abstract"])
            })

paper_embeddings = np.vstack(paper_embeddings)
paper_embeddings /= (np.linalg.norm(paper_embeddings, axis=1, keepdims=True) + 1e-12)
paper_sciBERT = np.vstack(paper_sciBERT)
paper_sciBERT /= (np.linalg.norm(paper_sciBERT, axis=1, keepdims=True) + 1e-12)

# ─── Load SciBERT ────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to("cuda")
model.eval()

# ─── Load Projector ──────────────────────────────────────────────────────
class Projector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1536, 128)
    def forward(self, x):
        return self.linear(x)

projector = Projector()
projector.load_state_dict(torch.load(projector_path, map_location="cuda"))
projector.eval().to("cuda")

# ─── Helper Functions ────────────────────────────────────────────────────
def embed_paragraph(text):
    if not isinstance(text, str) or not text.strip():
        return np.zeros(1536, dtype=np.float32)
    with torch.no_grad():
        inputs = tokenizer(text, padding="max_length", max_length=512,
                           truncation=True, return_tensors="pt").to("cuda")
        outputs = model(**inputs)
        cls = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
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
        references = [str(r) for r in obj.get("references", [])]

        query_vec = embed_paragraph(para)
        query_vec /= np.linalg.norm(query_vec) + 1e-12

        # Projection
        query_tensor = torch.tensor(query_vec, dtype=torch.float32).unsqueeze(0).to("cuda")
        query_proj = projector(query_tensor).squeeze(0).detach().cpu().numpy()
        query_proj /= np.linalg.norm(query_proj) + 1e-12

        # Combine
        semantic_sim = paper_sciBERT @ query_vec
        structural_sim = paper_embeddings @ query_proj
        final_scores = 0.8 * semantic_sim + 0.2 * structural_sim
        top_k = 5
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        top_pred_ids = [paper_records[i]["id"] for i in top_indices]

        references = [r for r in references if r in paper_id_map]
        if references:
            metrics = compute_metrics(top_pred_ids, references)
            results.append(metrics)

# ─── Print Results ───────────────────────────────────────────────────────
results = np.array(results)
avg_precision, avg_recall, avg_hr, avg_ndcg = results.mean(axis=0)
metrics = {
    "Precision@5": round(avg_precision, 4),
    "Recall@5": round(avg_recall, 4),
    "HR@5": round(avg_hr, 4),
    "NDCG@5": round(avg_ndcg, 4),
    "Evaluated Samples": len(results)
}

metrics
