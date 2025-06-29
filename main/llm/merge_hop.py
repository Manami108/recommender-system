import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path("./csv/metrics_hop_sweep.csv")
df = pd.read_csv(csv_path)

# ── reshape ─────────────────────────────────────────────
melted = (
    df.melt(id_vars=["hop_n"], var_name="metric_at_k", value_name="value")
      .assign(**{
          "metric": lambda d: d.metric_at_k.str.split("@").str[0],
          "k":      lambda d: d.metric_at_k.str.split("@").str[1].astype(int)
      })
)

# ── plot: one figure per metric, curves = hop_n ─────────
save_dir = Path("./eval"); save_dir.mkdir(exist_ok=True)

for metric in ["P", "HR", "R", "NDCG"]:
    pivot = (
        melted[melted.metric == metric]
        .pivot(index="k", columns="hop_n", values="value")
        .sort_index()                           # k in order 3,5,10,20
    )

    plt.figure()
    pivot.plot(marker="o", ms=4, ax=plt.gca())  # one line per hop_n
    plt.title(f"{metric}@k vs k  (lines = hop_n)")
    plt.xlabel("k  (# recommendations)")
    plt.ylabel(metric)                        
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / f"{metric.lower()}_all_hops.png", dpi=200)
    plt.close()
