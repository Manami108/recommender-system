import pandas as pd
import matplotlib.pyplot as plt

# 1) load all CSVs
files = [
    "metrics_rrf_llm_working1.csv",
    "metrics_rrf_llm_working2.csv",
    # "metrics_rrf_llm_working3.csv",
    "metrics_rrf_llm.csv",
]
dfs = [pd.read_csv(f"./csv/{fn}") for fn in files]

# 2) concatenate
all_df = pd.concat(dfs, ignore_index=True)

# 3) melt so you have columns: method, metric, k, value
melted = all_df.melt(                     # ← id_vars changed
    id_vars=["method"],
    var_name="metric_at_k",
    value_name="value"
)

melted[["metric", "k"]] = melted["metric_at_k"].str.split("@", expand=True)
melted["k"] = melted["k"].astype(int)

# 4) pivot so rows are k, columns are methods, values are average precision
for metric in ["P", "R", "HR", "NDCG"]:
    dfm = (
        melted[melted.metric == metric]
        .groupby(["k", "method"])["value"]
        .mean()
        .reset_index()
        .pivot(index="k", columns="method", values="value")   # ← columns changed
    )
    plt.figure()
    dfm.plot(marker="o", markersize=4, ax=plt.gca())
    plt.title(f"{metric}@k comparison")
    plt.xlabel("k")
    plt.ylabel(metric)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./eval/{metric.lower()}_all_prompts.png", dpi=200)
    plt.close()
