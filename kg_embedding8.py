# complEx is used for this code. MINERVA is better ig.


import os
import pandas as pd
import torch
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

# =============================== Configuration ===============================
# Paths
graph_nodes_path = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/graph_nodes.csv"
graph_edges_path = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/graph_edges_clean.csv"
output_dir = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/kge_compgcn_model"

# Model parameters
model_name = "CompGCN"
embedding_dim = 200
num_layers = 2
learning_rate = 1e-3
num_epochs = 100
random_seed = 42

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================== Load Edges ===============================
print("[Step 1/5] Loading edges...")
if not os.path.exists(graph_edges_path):
    raise FileNotFoundError(f"Graph edges file not found: {graph_edges_path}")

df_edges = pd.read_csv(graph_edges_path)

required_cols = [":START_ID", ":END_ID", ":TYPE"]
if not all(col in df_edges.columns for col in required_cols):
    raise ValueError(f"Missing required columns in edges file: {required_cols}")

# Drop rows with missing or invalid relations
df_edges = df_edges.dropna(subset=[":START_ID", ":END_ID", ":TYPE"])
df_edges[":TYPE"] = df_edges[":TYPE"].astype(str).str.strip()
df_edges[":START_ID"] = df_edges[":START_ID"].astype(str).str.strip()
df_edges[":END_ID"] = df_edges[":END_ID"].astype(str).str.strip()

print(f"Total triples after cleaning: {len(df_edges)}")

triples = df_edges[[":START_ID", ":TYPE", ":END_ID"]].values

# ==================== Build TriplesFactory ====================
print("[Step 2/5] Building TriplesFactory...")
triples_factory = TriplesFactory.from_labeled_triples(
    triples,
    create_inverse_triples=True
)

result = pipeline(
    training=triples_factory,
    testing=triples_factory,
    model="ComplEx",
    model_kwargs={
        "embedding_dim": 200,
    },
    optimizer="Adam",
    optimizer_kwargs={"lr": 1e-3},
    training_kwargs={
        "num_epochs": 100,
        "slice_size": 256,
        "sub_batch_size": 512,
    },
    evaluator="rankbased",
    random_seed=42,
    device="cpu",   

# =============================== Save Model ===============================
print("[Step 4/5] Saving model artifacts...")
os.makedirs(output_dir, exist_ok=True)
result.save_to_directory(output_dir)

print(f"[Step 5/5] Model saved to {output_dir}")
print("Finished training and saving CompGCN Knowledge Graph Embedding!")
