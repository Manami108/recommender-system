import pandas as pd
import os

# =============================== Paths ===============================
graph_edges_path = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/graph_edges_clean.csv"
minerva_triples_path = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/MINERVA8-3/datasets/minerva_triples.tsv"

# =============================== Load Edges ===============================
print("Loading edges...")
df_edges = pd.read_csv(graph_edges_path)

# Check necessary columns
required_cols = [":START_ID", ":END_ID", ":TYPE"]
if not all(col in df_edges.columns for col in required_cols):
    raise ValueError(f"Missing required columns: {required_cols}")

# Drop rows with missing data
df_edges = df_edges.dropna(subset=[":START_ID", ":END_ID", ":TYPE"])

# Convert to string and clean up spaces
df_edges[":START_ID"] = df_edges[":START_ID"].astype(str).str.strip()
df_edges[":END_ID"] = df_edges[":END_ID"].astype(str).str.strip()
df_edges[":TYPE"] = df_edges[":TYPE"].astype(str).str.strip()

# =============================== Save in MINERVA format ===============================
print(f"Saving triples to {minerva_triples_path}...")

# Save as tab-separated file
df_edges[[":START_ID", ":TYPE", ":END_ID"]].to_csv(
    minerva_triples_path,
    sep="\t",
    index=False,
    header=False
)

print(f"Finished saving {len(df_edges)} triples in MINERVA format!")
