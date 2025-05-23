#!/usr/bin/env python
# node2vec for graph embedding 
# batch_words=1024
# walk lengthe is set to 80

import argparse, csv, numpy as np, pandas as pd, networkx as nx
from tqdm import tqdm
from node2vec import Node2Vec

def vec_to_str(v): return ";".join(f"{x:.5f}" for x in v)

def main():
    # need to run with this
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges",   required=True, help="CSV with two columns: source,target")
    ap.add_argument("--out",     required=True, help="Output CSV with Node2Vec embeddings")
    ap.add_argument("--dims",    type=int, default=128)
    ap.add_argument("--walks",   type=int, default=20)
    ap.add_argument("--window",  type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    # load edge list 
    print("Loading edge list …")
    df = pd.read_csv(args.edges, header=None, dtype={0: str, 1: str}, low_memory=False)

    src_col, dst_col = df.columns[:2]
    df[src_col] = df[src_col].astype(str)
    df[dst_col] = df[dst_col].astype(str)

    G = nx.from_pandas_edgelist(df, source=src_col, target=dst_col,
                                create_using=nx.DiGraph())
    print(f"  Papers: {G.number_of_nodes():,}  CITES edges: {G.number_of_edges():,}")

    # train node2vec
    print("Training Node2Vec …")
    node2vec = Node2Vec(G,
                        dimensions=args.dims,
                        walk_length=80,
                        num_walks=args.walks,
                        workers=args.workers,
                        quiet=True)
    model = node2vec.fit(window=args.window, min_count=1, batch_words=1024)

    # write embedding 
    zero_vec = vec_to_str(np.zeros(args.dims, dtype=np.float32))
    print("Writing vectors …")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["id:ID(Paper)", "n2v:float[]"])
        for node in tqdm(G.nodes(), total=G.number_of_nodes()):
            key = str(node)
            vec = vec_to_str(model.wv[key]) if key in model.wv else zero_vec
            w.writerow([key, vec])

    print(f"Saved Node2Vec embeddings → {args.out}")

if __name__ == "__main__":
    main()


#  python knowledgeEmbedding.py --edges /media/e-soc-student/DISK2/GR/GR2_Recommendation/export_csv/paper_cites_paper.csv --out /media/e-soc-student/DISK2/GR/GR2_Recommendation/export_csv/paper_n2v.csv --dims 128 --walks 10 --window 5 --workers 4
