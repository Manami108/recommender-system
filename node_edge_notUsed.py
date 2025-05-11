import pandas as pd

# File paths (adjust as needed)
node_file = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/node_papers.csv"
edge_file = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/edge_citations.csv"
orphan_file = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/orphan_nodes.csv"

# Load node CSV (ensure IDs are strings and stripped)
nodes_df = pd.read_csv(node_file, dtype=str)
nodes_df['paperId:ID'] = nodes_df['paperId:ID'].astype(str).str.strip()
valid_ids = set(nodes_df['paperId:ID'])

# Load relationships CSV (ensure IDs are strings and stripped)
edges_df = pd.read_csv(edge_file, dtype=str)
edges_df[':START_ID'] = edges_df[':START_ID'].astype(str).str.strip()
edges_df[':END_ID'] = edges_df[':END_ID'].astype(str).str.strip()

# Get all IDs referenced in relationships
ref_ids = set(edges_df[':START_ID']).union(set(edges_df[':END_ID']))

# Find missing IDs: those referenced in relationships but not in valid_ids
missing_ids = ref_ids - valid_ids

print(f"Total valid node IDs: {len(valid_ids)}")
print(f"Total referenced IDs in relationships: {len(ref_ids)}")
print(f"Missing (orphan) IDs: {len(missing_ids)}")

# Create a DataFrame for orphan nodes.
# For these nodes, we only have the ID; you can leave other fields blank.
orphan_df = pd.DataFrame(list(missing_ids), columns=["paperId:ID"])
orphan_df[":LABEL"] = "Paper"
# Optionally, add empty columns for title, year, n_citation, fos so that both CSVs have the same structure
orphan_df["title"] = ""
orphan_df["year"] = ""
orphan_df["n_citation"] = ""
orphan_df["fos"] = ""

# Ensure the column order matches the node CSV: paperId:ID,title,year,n_citation,fos,:LABEL
orphan_df = orphan_df[["paperId:ID", "title", "year", "n_citation", "fos", ":LABEL"]]

# Save the orphan nodes CSV
orphan_df.to_csv(orphan_file, index=False)
print(f"Orphan nodes CSV saved to {orphan_file} with {orphan_df.shape[0]} records.")
