# Fos embedding is done to include in neo4j node. 
# SciBERT is used for embedding, cosign similarity is normalized 
# the dimention is 768
# Save as numpy arrey


import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import ast
import re
import time
import json

# Config 
csv_path = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/small_dblp.v12.csv"
output_embedding_path = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/small_fos_embeddings_sciBERT.npy"
output_fos_list_path = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/small_fos_list.txt"
output_fos_index_path = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/small_fos_index.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset 
print("\u25b6 Loading CSV...")
df = pd.read_csv(csv_path, usecols=["fos"], low_memory=False)
df["fos"] = df["fos"].fillna("[]")

# Parse Unique FOS Terms 
def safe_eval(s: str):
    try:
        if isinstance(s, str):
            s = re.sub(r"Decimal\('([^']+)'\)", r"\1", s)
            out = ast.literal_eval(s)
            return out if isinstance(out, list) else []
        return []
    except Exception:
        return []

print("\u25b6 Extracting unique FOS terms...")
df["fos_parsed"] = df["fos"].apply(safe_eval)

fos_set = set()
for fos_list in df["fos_parsed"]:
    for f in fos_list:
        if isinstance(f, dict) and "name" in f:
            fos_set.add(f["name"].strip().lower())  
        elif isinstance(f, str):
            fos_set.add(f.strip().lower())

fos_list = sorted(fos_set)
print(f"✓ Total unique FOS terms: {len(fos_list):,}")

# Load SciBERT 
print("\u25b6 Loading SciBERT model...")
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(device)
model.eval()
print("\u2713 SciBERT ready")

# Embedding Function 
def generate_embedding(text):
    if not isinstance(text, str) or not text.strip():
        return np.zeros(768, dtype=np.float32)
    try:
        inputs = tokenizer(
            text,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        vec = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy().astype(np.float32)
        # ─── normalize for cosine similarity ───
        vec /= (np.linalg.norm(vec) + 1e-12)
        return vec
    except Exception:
        return np.zeros(768, dtype=np.float32)


# Embedding Loop 
print("\u25b6 Embedding FOS terms...")
start_time = time.time()
fos_embeddings = []

for fos in tqdm(fos_list):
    emb = generate_embedding(fos)
    fos_embeddings.append(emb)

fos_embeddings = np.array(fos_embeddings, dtype=np.float32)

# Save Outputs 
np.save(output_embedding_path, fos_embeddings)
with open(output_fos_list_path, "w", encoding="utf-8") as f:
    f.writelines([f"{term}\n" for term in fos_list])

fos_id2idx = {term: idx for idx, term in enumerate(fos_list)}
with open(output_fos_index_path, "w", encoding="utf-8") as f:
    json.dump(fos_id2idx, f, ensure_ascii=False, indent=2)

print(f"✓ FOS embeddings saved to: {output_embedding_path}")
print(f"✓ FOS list saved to      : {output_fos_list_path}")
print(f"✓ FOS index mapping saved: {output_fos_index_path}")
print(f"Total time: {time.time() - start_time:.2f} seconds")