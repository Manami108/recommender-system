import os
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURATION ===
INCLUDE_METHODS = ["metrics_rrf_llm_working32", "metrics_rrf_hop3_llm", "metrics_rrf_hop1_llm"]  # ← just fill in the ones you want

base_dir      = "/home/abhi/Desktop/Manami/recommender-system/main/llm"
csv_final_dir = os.path.join(base_dir, "csv_final")
eval_dir      = os.path.join(base_dir, "eval_final")

os.makedirs(eval_dir, exist_ok=True)

# Gather all merged CSVs
dfs = []
for fname in os.listdir(csv_final_dir):
    if not fname.endswith(".csv"):
        continue
    path = os.path.join(csv_final_dir, fname)
    method = os.path.splitext(fname)[0]
    df = pd.read_csv(path)
    df["method"] = method
    dfs.append(df)

if not dfs:
    raise RuntimeError("No merged CSVs found in " + csv_final_dir)

# Concatenate all methods
all_df = pd.concat(dfs, ignore_index=True)
all_df = all_df[all_df["method"].isin(INCLUDE_METHODS)]

# Melt & split metric and k
melted = all_df.melt(
    id_vars=["method"],
    var_name="metric_at_k",
    value_name="value"
)
melted = melted.dropna(subset=["metric_at_k"])
melted[["metric", "k"]] = melted["metric_at_k"].str.split("@", expand=True)
melted = melted.dropna(subset=["k"])
melted["k"] = melted["k"].astype(int)

# Plot each metric comparing the methods
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
    plt.title(f"{metric}@k Comparison")
    plt.xlabel("k")
    plt.ylabel(metric)
    plt.grid(True)
    plt.tight_layout()
    out_file = os.path.join(eval_dir, f"{metric.lower()}_comparison_hop.png")
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved plot: {out_file}")
