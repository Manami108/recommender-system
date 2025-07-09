import os
import pandas as pd

# === CONFIGURATION ===
base_dir     = "/home/abhi/Desktop/Manami/recommender-system/main/llm"
csv_dirs     = [os.path.join(base_dir, f"csv{i}") for i in range(1, 5)]
csv_final_dir = os.path.join(base_dir, "csv_final")

# Ensure output directory exists
os.makedirs(csv_final_dir, exist_ok=True)

# List of suffixes (final filenames) to merge
suffixes = [
    "metrics_rrf_hop3_llm.csv", 
]

# Merge per suffix across csv1-4, drop failures, then save
for suffix in suffixes:
    dfs = []
    for i, d in enumerate(csv_dirs, start=1):
        path = os.path.join(d, f"{i}{suffix}")
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
        else:
            print(f"Warning: {path} not found")

    if not dfs:
        print(f"No files found for {suffix}, skipping.")
        continue

    merged = pd.concat(dfs, ignore_index=True)
    print(merged["rerank_failed"].dtype)
    print(merged["rerank_failed"].unique())


    # Drop rerank failures if that column exists
    # Drop rerank failures if that column exists
    if "rerank_failed" in merged.columns:
        before = len(merged)
        mask = merged["rerank_failed"].fillna(False).astype(bool)
        merged = merged[~mask]
        print(f"{suffix}: Dropped {before - len(merged)} failed rows")


    out_path = os.path.join(csv_final_dir, suffix)
    merged.to_csv(out_path, index=False)
    print(f"Saved merged CSV to {out_path}")
