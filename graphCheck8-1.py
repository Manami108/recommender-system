import pandas as pd

edge_file = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/graph_edges_clean.csv"

print("Loading edges...")
df_edges = pd.read_csv(edge_file)

# Check if essential columns exist
required_cols = [":START_ID", ":END_ID", ":TYPE"]
if not all(col in df_edges.columns for col in required_cols):
    raise ValueError(f"Missing required columns: {required_cols}")

print(f"Total edges loaded: {len(df_edges)}")

# Check missing values
missing_start = df_edges[":START_ID"].isnull().sum()
missing_end = df_edges[":END_ID"].isnull().sum()
missing_type = df_edges[":TYPE"].isnull().sum()

print(f"Missing :START_ID: {missing_start}")
print(f"Missing :END_ID: {missing_end}")
print(f"Missing :TYPE: {missing_type}")

# Show how many different types
print("Relation types found:")
print(df_edges[":TYPE"].value_counts())

# Preview the first few rows
print("\nSample edges:")
print(df_edges.head())
