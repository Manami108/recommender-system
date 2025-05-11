import hnswlib
import numpy as np
import pandas as pd
import time

# Paths
embeddings_file = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/embeddings_sciBERT.npy"
df_file = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/preprocessed_dblp.csv"
similarity_file = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/similarity_top10.csv"
index_path = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/hnsw_index.bin"

start_time = time.time()

# Load data
df = pd.read_csv(df_file, low_memory=False)
embeddings = np.load(embeddings_file).astype(np.float32)

# Normalize embeddings
embeddings /= (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

num_embeddings, dim = embeddings.shape
print(f"Number of embeddings: {num_embeddings}, Dimension: {dim}")

# Initialize HNSW index
index = hnswlib.Index(space='cosine', dim=dim)
index.init_index(max_elements=num_embeddings, ef_construction=200, M=16)
index.add_items(embeddings)
index.set_ef(50)  # you can try 100 for better recall

# Save index for reuse
index.save_index(index_path)
print(f"HNSW index saved to: {index_path}")

# Search for top-k
top_k = 10
print(f"Searching top-{top_k} neighbors for all vectors...")
search_start = time.time()
distances, neighbors = index.knn_query(embeddings, k=top_k)
print(f"Search done in {time.time() - search_start:.2f} seconds")

# Build results
results = []
for i, (dist_row, idx_row) in enumerate(zip(distances, neighbors)):
    for dist, nn_idx in zip(dist_row, idx_row):
        if i != nn_idx:  # skip self
            cos_sim = 1.0 - dist
            results.append({
                "paper1_id": df['id'].iloc[i],
                "paper2_id": df['id'].iloc[int(nn_idx)],
                "similarity_score": cos_sim
            })

# Save to CSV
similarity_df = pd.DataFrame(results)
similarity_df.to_csv(similarity_file, index=False)
print(f"Results saved to: {similarity_file}")
print(f"Total time: {time.time() - start_time:.2f} seconds")
