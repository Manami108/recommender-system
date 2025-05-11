import pandas as pd
import ast

# Load the original pure DBLP dataset
pure_dblp_path = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/dblp.v12.csv"
output_path = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/preprocessed_dblp_FIXED.csv"

print("Loading pure DBLP...")
df = pd.read_csv(pure_dblp_path)

print("Fixing references...")
# Fill missing references with "[]", but DO NOT overwrite good ones
df['references'] = df['references'].fillna("[]")

# Optionally, strip .0 from IDs if needed (optional)
df['id'] = df['id'].astype(str).str.replace(r'\.0$', '', regex=True)

# Save the corrected dataset
df.to_csv(output_path, index=False)

print(f"Saved corrected preprocessed dataset to: {output_path}")
