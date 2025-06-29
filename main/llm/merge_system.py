import pandas as pd
import matplotlib.pyplot as plt

# 1) load all CSVs
files = [
    "metrics_bm25_full.csv",
    "metrics_bm25_full_llm.csv",
    "metrics_rrf.csv",
    "metrics_rrf_llm.csv",
    # "metrics_rrf_hop_llm.csv",
]
dfs = [pd.read_csv(f"./csv/{fn}") for fn in files]

# 2) concatenate
all_df = pd.concat(dfs, ignore_index=True)

# 3) compute average per method & k
#    melt so you have columns: method, metric, k, value
melted = all_df.melt(
    id_vars=["method"], 
    var_name="metric_at_k", 
    value_name="value"
)

# split metric_at_k into metric and k
melted[["metric","k"]] = melted["metric_at_k"].str.split("@", expand=True)
melted["k"] = melted["k"].astype(int)

# 4) pivot so rows are k, columns are methods, values are average precision
for metric in ["P","R","HR","NDCG"]:
    dfm = (
        melted[melted.metric==metric]
        .groupby(["k","method"])["value"]
        .mean()
        .reset_index()
        .pivot(index="k", columns="method", values="value")
    )
    plt.figure()
    dfm.plot(marker="o", ax=plt.gca())
    plt.title(f"{metric}@k comparison")
    plt.xlabel("k")
    plt.ylabel(metric)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./eval/{metric.lower()}_all_methods.png", dpi=200)
    plt.close()



