import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import time
from tqdm import tqdm

# =============================================================================
# Step 1: Load Dataset
# =============================================================================
preprocessed_csv = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/preprocessed_dblp.csv"
print("Loading preprocessed dataset...")
start_time = time.time()
df = pd.read_csv(preprocessed_csv, low_memory=False)

# Verify required columns
required_cols = ['title', 'indexed_abstract']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")
print(f"Dataset loaded. Total papers: {len(df)}")

# =============================================================================
# Step 2: Load SciBERT Model & Tokenizer
# =============================================================================
print("Loading SciBERT model...")
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model_text = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to("cuda")
model_text.eval()
print("SciBERT model loaded.")

# =============================================================================
# Step 3: Embedding Function (using [CLS] token)
# =============================================================================
def generate_embedding(text):
    if not isinstance(text, str) or not text.strip():
        return np.zeros(768, dtype=np.float32)
    try:
        inputs = tokenizer(text, padding="max_length", max_length=512,
                           truncation=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model_text(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy().astype(np.float32)
    except Exception:
        return np.zeros(768, dtype=np.float32)

# =============================================================================
# Step 4: Combine Title + Abstract Embeddings
# =============================================================================
def generate_text_embedding(title, abstract):
    title_emb = generate_embedding(title)
    abstract_emb = generate_embedding(abstract)
    return np.concatenate([title_emb, abstract_emb]).astype(np.float32)  # 1536-dim

# =============================================================================
# Step 5: Batch Processing and Save
# =============================================================================
batch_size = 512
text_embeddings = []

print("Generating SciBERT embeddings for title + abstract...")
for i in tqdm(range(0, len(df), batch_size)):
    titles = df['title'][i:i + batch_size].tolist()
    abstracts = df['indexed_abstract'][i:i + batch_size].tolist()

    batch_embeds = [
        generate_text_embedding(title, abstract)
        for title, abstract in zip(titles, abstracts)
    ]
    text_embeddings.extend(batch_embeds)

# Save embeddings
output_file = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/embeddings_sciBERT.npy"
text_embeddings = np.array(text_embeddings, dtype=np.float32)
np.save(output_file, text_embeddings)

print(f"Embeddings saved to {output_file}")
print(f"Total time: {time.time() - start_time:.2f} seconds")
