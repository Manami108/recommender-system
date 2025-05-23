
# Fos embedding is done to include in neo4j node. 
# SciBERT is used for embedding, cosign similarity is normalized 
# the dimention is 768
# Save as numpy arrey
# title + abstract
# Abstract is reconstracted
# Batch size is set to 512

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch, time, ast
from tqdm import tqdm

# load the dataset
CSV_PATH = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/small/small_dblp.v12.csv"
print("Loading preprocessed dataset...")
t0 = time.time()
df = pd.read_csv(CSV_PATH, low_memory=False)

# minimal sanity checks
if "title" not in df.columns:
    raise ValueError("Missing required column: title")
if "indexed_abstract" not in df.columns:
    print("Warning: 'indexed_abstract' column missing. Proceeding with titles only.")
    df["indexed_abstract"] = ""

print(f"Dataset loaded. Total papers: {len(df):,}")

# SciBERT
print("Loading SciBERT model...")
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model     = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to("cuda").eval()
print("SciBERT ready.")

# Recnstract abstract
def reconstruct_abstract(indexed_abstract_str: str) -> str:
    """Turn DBLP's indexed_abstract JSON back into plain text."""
    if not isinstance(indexed_abstract_str, str) or not indexed_abstract_str.strip():
        return ""
    try:
        obj       = ast.literal_eval(indexed_abstract_str)
        length    = obj.get("IndexLength", 0)
        inv_index = obj.get("InvertedIndex", {})
        words     = [""] * length
        for w, positions in inv_index.items():
            for p in positions:
                if 0 <= p < length:
                    words[p] = w
        return " ".join(words).strip()
    except Exception:
        return ""

@torch.no_grad()
def encode_text(text: str) -> np.ndarray:
    """CLS-token representation (float32, 768-d)."""
    if not text.strip():
        return np.zeros(768, dtype=np.float32)
    inputs = tokenizer(
        text, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
    ).to("cuda")
    outputs = model(**inputs)
    cls_vec = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy().astype(np.float32)
    # ─── NEW ─── normalize for cosine
    cls_vec /= (np.linalg.norm(cls_vec) + 1e-12)
    return cls_vec

# embedding 
BATCH_SIZE  = 512
embeddings  = []

print("Encoding papers with SciBERT (title + abstract)…")
for start in tqdm(range(0, len(df), BATCH_SIZE)):
    batch_df = df.iloc[start : start + BATCH_SIZE]
    for title, ia in zip(batch_df["title"], batch_df["indexed_abstract"]):
        abstract_txt   = reconstruct_abstract(ia)
        combined_text  = f"{title.strip()}. {abstract_txt}" if abstract_txt else title.strip()
        vec            = encode_text(combined_text)
        embeddings.append(vec)

embeddings = np.stack(embeddings, dtype=np.float32)   # shape: (N, 768)

# ─────────── 5. Save ──────────────────────────
OUT_PATH = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/small/small_embeddings_sciBERT.npy"
np.save(OUT_PATH, embeddings)

print(f"\nEmbeddings saved → {OUT_PATH}")
print(f"Total vectors: {embeddings.shape[0]:,}  |  Dim: {embeddings.shape[1]}")
print(f"Elapsed: {time.time() - t0:.1f} s")
