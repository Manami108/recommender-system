

# Fos embedding is done to include in neo4j node. 
# SciBERT is used for embedding, cosign similarity is normalized 
# Abstract is reconstracted
# hnsw indexing but currently, not used

import hnswlib
import numpy as np
import pandas as pd
import time

EMB_FILE   = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/small/small_embeddings_sciBERT.npy"
CSV_FILE   = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/small/small_dblp.v12.csv"
INDEX_FILE = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/small/small_hnsw_index.bin"

t0 = time.time()

# data loading 
df         = pd.read_csv(CSV_FILE, low_memory=False)
embeddings = np.load(EMB_FILE).astype(np.float32)

# Safety check
assert len(df) == len(embeddings), "CSV rows ≠ embeddings rows!"

# Normalize for cosine similarity
embeddings /= (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

num, dim = embeddings.shape
print(f"Loaded {num:,} embeddings  |  dim = {dim}")

# hnsw indexing 
print("Building HNSW index…")
index = hnswlib.Index(space="cosine", dim=dim)
index.init_index(max_elements=num, ef_construction=200, M=16)

labels = np.arange(num, dtype=np.int32)          # 0…N-1
index.add_items(embeddings, labels)
index.set_ef(50)                                 # runtime search parameter

index.save_index(INDEX_FILE)
print(f"Index saved → {INDEX_FILE}")
print(f"Elapsed: {time.time() - t0:.1f} s")

# ────────────── Similarity search (optional) ─────────────────
# Uncomment this block only if you later need the top-k CSV
# -------------------------------------------------------------
# TOP_K      = 10
# SIM_FILE   = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/big/similarity_top10.csv"
#
# print(f"Searching top-{TOP_K} neighbors for all vectors…")
# search_start = time.time()
# neighbors, distances = index.knn_query(embeddings, k=TOP_K)
# print(f"Search done in {time.time() - search_start:.2f} s")
#
# results = []
# for i, (dists, idxs) in enumerate(zip(distances, neighbors)):
#     for dist, j in zip(dists, idxs):
#         if i == j:          # skip self-match
#             continue
#         results.append({
#             "paper1_id": df.iloc[i]["id"],
#             "paper2_id": df.iloc[j]["id"],
#             "similarity_score": 1.0 - dist      # cosine sim
#         })
#
# pd.DataFrame(results).to_csv(SIM_FILE, index=False)
# print(f"Similarities saved → {SIM_FILE}")
# -------------------------------------------------------------
