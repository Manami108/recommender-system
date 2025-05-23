#!/usr/bin/env python
# need to merge the graph embedding to paper nodes
import argparse, pandas as pd, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="paper_nodes.csv")
    ap.add_argument("--n2v",  required=True, help="paper_n2v.csv")
    ap.add_argument("--out",  required=True, help="paper_nodes_merged.csv")
    args = ap.parse_args()

    print("Reading base nodes …")
    df_nodes = pd.read_csv(args.base, dtype={"id:ID(Paper)": str})
    print("  rows:", len(df_nodes))

    print("Reading n2v …")
    df_n2v = pd.read_csv(args.n2v, dtype={"id:ID(Paper)": str})
    print("  rows:", len(df_n2v))

    print("Merging …")
    merged = df_nodes.merge(df_n2v, on="id:ID(Paper)", how="left")
    # fill missing vectors (isolated nodes) with zeros
    dims = len(df_n2v.iloc[0, 1].split(";"))
    merged["n2v:float[]"] = merged["n2v:float[]"].fillna(";".join(["0.0"]*dims))

    print("Writing", args.out)
    merged.to_csv(args.out, index=False)
    print("Done. Final rows:", len(merged))

if __name__ == "__main__":
    main()


#  python merge.py --base /media/e-soc-student/DISK2/GR/GR2_Recommendation/export_csv/paper_nodes.csv --n2v /media/e-soc-student/DISK2/GR/GR2_Recommendation/export_csv/paper_n2v.csv --out /media/e-soc-student/DISK2/GR/GR2_Recommendation/export_csv/paper_nodes_merged.csv
