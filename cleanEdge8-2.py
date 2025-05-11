import pandas as pd

edge_file = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/graph_edges.csv"
output_clean_edge_file = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/graph_edges_clean.csv"

print("Loading edges...")
df_edges = pd.read_csv(edge_file)

# Drop rows with missing IDs
print(f"Original number of edges: {len(df_edges)}")
df_edges = df_edges.dropna(subset=[":START_ID", ":END_ID", ":TYPE"])
print(f"After dropping missing IDs: {len(df_edges)}")

# Convert IDs to string and remove decimal points
df_edges[":START_ID"] = df_edges[":START_ID"].astype(str).str.replace(r'\.0$', '', regex=True)
df_edges[":END_ID"] = df_edges[":END_ID"].astype(str).str.replace(r'\.0$', '', regex=True)

# Check unique relation types
print("Relation types found after cleaning:")
print(df_edges[":TYPE"].value_counts())

# Save cleaned edges
df_edges.to_csv(output_clean_edge_file, index=False)
print(f"Cleaned edges saved to {output_clean_edge_file}")
