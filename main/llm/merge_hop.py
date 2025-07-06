import pandas as pd
import matplotlib.pyplot as plt

# 1) load all CSVs
files = [
    "1metrics_rrf_hop1_llm.csv",
    "1metrics_rrf_llm_working32.csv",
    "metrics_rrf_pure_llama.csv",
    # "metrics_rrf_hop10_llm.csv",
]
dfs = [pd.read_csv(f"./csv1/{fn}") for fn in files]

# 2) concatenate
all_df = pd.concat(dfs, ignore_index=True)

# 3) melt so you have columns: method, metric_at_k, value
melted = all_df.melt(
    id_vars=["method"],
    var_name="metric_at_k",
    value_name="value"
)

# split metric_at_k into metric and k; coerce non-numeric k's to NaN
split = melted["metric_at_k"].str.split("@", expand=True)
melted["metric"] = split[0]
melted["k"] = pd.to_numeric(split[1], errors="coerce")

# drop any rows where k couldn't be parsed
melted = melted.dropna(subset=["k"])
melted["k"] = melted["k"].astype(int)

# 4) pivot & plot
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
    plt.title(f"{metric}@k comparison")
    plt.xlabel("k")
    plt.ylabel(metric)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./eval/{metric.lower()}_all_hops.png", dpi=200)
    plt.close()
