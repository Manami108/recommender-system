import os
import pandas as pd
import matplotlib.pyplot as plt

# Directories
base_dir = "/home/abhi/Desktop/Manami/recommender-system/main/llm"
csv_dirs = [os.path.join(base_dir, f"csv{i}") for i in range(1, 5)]
csv_final_dir = os.path.join(base_dir, "csv_final")
eval_dir = os.path.join(base_dir, "eval_final")

# Ensure output directories exist
os.makedirs(csv_final_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

# List of suffixes (final filenames) to merge
suffixes = [
    "metrics_bm25.csv",
    "metrics_rrf.csv",
    "metrics_rrf_llm_working22.csv",
    "metrics_rrf_llm_working32.csv",
]

# 1) Merge per suffix across csv1-4 and save to csv_final
for suffix in suffixes:
    dfs = []
    for i, d in enumerate(csv_dirs, start=1):
        fname = f"{i}{suffix}"
        path = os.path.join(d, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            dfs.append(df)
        else:
            print(f"Warning: {path} not found")
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        out_path = os.path.join(csv_final_dir, suffix)
        merged.to_csv(out_path, index=False)

# 2) Plot comparison of the four merged results
#    Read each merged CSV, label by method (suffix without extension)
dfs = []
for suffix in suffixes:
    path = os.path.join(csv_final_dir, suffix)
    method = os.path.splitext(suffix)[0]
    df = pd.read_csv(path)
    df["method"] = method
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

# Melt & split metric and k
melted = all_df.melt(
    id_vars=["method"],
    var_name="metric_at_k",
    value_name="value"
)
melted[["metric", "k"]] = melted["metric_at_k"].str.split("@", expand=True)
melted = melted.dropna(subset=["k"])
melted["k"] = melted["k"].astype(int)

# Plot each metric comparing the four methods
for metric in ["P", "R", "HR", "NDCG"]:
    dfm = (
        melted[melted.metric == metric]
        .groupby(["k", "method"])["value"]
        .mean()
        .reset_index()
        .pivot(index="k", columns="method", values="value")
    )
    plt.figure()
    dfm.plot(marker="o", markersize=4, ax=plt.gca())
    plt.title(f"{metric}@k comparison (final merged)")
    plt.xlabel("k")
    plt.ylabel(metric)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, f"{metric.lower()}_final_comparison.png"), dpi=200)
    plt.close()
