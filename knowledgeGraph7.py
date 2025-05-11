import os
import re
import ast
import argparse
from typing import List, Dict

import dgl
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def safe_eval(s: str):
    """Safely evaluate stringified Python literals (list / dict) and handle Decimal."""
    try:
        if isinstance(s, str):
            s = re.sub(r"Decimal\('([^']+)'\)", r"\1", s)
            out = ast.literal_eval(s)
            return out if isinstance(out, list) else []
        return []
    except Exception:
        return []

def safe_tensor(src, dst):
    if len(src) == 0 or len(dst) == 0:
        return None
    return (torch.tensor(src, dtype=torch.int32), torch.tensor(dst, dtype=torch.int32))

def build_heterograph(csv_path: str, bertopic_path: str, output_graph: str, device: str = "cpu"):
    print("[1] Loading data ...")
    df = pd.read_csv(csv_path, low_memory=False)
    df["id"] = df["id"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    df["references"].fillna("[]", inplace=True)
    df["fos"].fillna("[]", inplace=True)
    df["title"].fillna("", inplace=True)
    df["indexed_abstract"].fillna("", inplace=True)

    paper_ids: List[str] = df["id"].tolist()
    paper_id2idx: Dict[str, int] = {pid: idx for idx, pid in enumerate(paper_ids)}
    print(f"   • Papers         : {len(paper_ids):,}")

    print("[2] Parsing FOS ...")
    df["fos_parsed"] = df["fos"].apply(safe_eval)

    fos_set = set()
    for fos_list in df["fos_parsed"]:
        for f in fos_list:
            if isinstance(f, dict) and "name" in f:
                fos_set.add(f["name"].strip())
            elif isinstance(f, str):
                fos_set.add(f.strip())
    fos_list = sorted(fos_set)
    fos_id2idx = {f: i for i, f in enumerate(fos_list)}
    print(f"   • FOS terms      : {len(fos_list):,}")

    print("[3] Loading BERTopic model ...")
    from bertopic import BERTopic
    topic_model = BERTopic.load(bertopic_path)
    topic_info = topic_model.get_topic_info()
    topic_labels = topic_info["Name"].tolist()
    topic_ids = topic_info["Topic"].astype(str).tolist()
    topic_id2idx = {tid: i for i, tid in enumerate(topic_ids)}
    print(f"   • Topics         : {len(topic_ids):,}")

    cites_src, cites_dst = [], []
    pf_src, pf_dst = [], []
    pt_src, pt_dst = [], []

    print("[4] Building edge lists ...")
    for row_idx, row in df.iterrows():
        pid_idx = row_idx

        for ref in safe_eval(row["references"]):
            ref = str(ref).replace(".0", "").strip()
            if ref in paper_id2idx:
                cites_src.append(pid_idx)
                cites_dst.append(paper_id2idx[ref])

        for f in row["fos_parsed"]:
            f_name = f["name"].strip() if isinstance(f, dict) else f.strip() if isinstance(f, str) else None
            if f_name and f_name in fos_id2idx:
                pf_src.append(pid_idx)
                pf_dst.append(fos_id2idx[f_name])

        topic_id = str(row.get("topic", -1))
        if topic_id != "-1" and topic_id in topic_id2idx:
            pt_src.append(pid_idx)
            pt_dst.append(topic_id2idx[topic_id])

    print("[5] Constructing DGL heterograph ...")
    data_dict = {}
    if safe_tensor(cites_src, cites_dst):
        data_dict[("paper", "cites", "paper")] = safe_tensor(cites_src, cites_dst)
    if safe_tensor(pf_src, pf_dst):
        data_dict[("paper", "has_fos", "fos")] = safe_tensor(pf_src, pf_dst)
        data_dict[("fos", "rev_has_fos", "paper")] = safe_tensor(pf_dst, pf_src)
    if safe_tensor(pt_src, pt_dst):
        data_dict[("paper", "has_topic", "topic")] = safe_tensor(pt_src, pt_dst)
        data_dict[("topic", "rev_has_topic", "paper")] = safe_tensor(pt_dst, pt_src)

    g = dgl.heterograph(data_dict)

    print("[6] Skipping assignment of string metadata to DGL. Saving externally ...")
    metadata_df = df[["id", "title", "indexed_abstract"]]
    metadata_path = output_graph.replace(".bin", "_paper_metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)
    print(f"✓ Paper metadata saved to {metadata_path}")

    print("[7] Embedding FOS & Topic labels ...")
    embed_model = SentenceTransformer("allenai/scibert_scivocab_uncased", device=device)
    fos_emb = embed_model.encode(fos_list, convert_to_tensor=True, show_progress_bar=True, batch_size=64)
    topic_emb = embed_model.encode(topic_labels, convert_to_tensor=True, show_progress_bar=True, batch_size=64)
    g.nodes["fos"].data["feat"] = fos_emb
    g.nodes["topic"].data["feat"] = topic_emb

    print("[8] Saving graph ...")
    os.makedirs(os.path.dirname(output_graph), exist_ok=True)
    dgl.save_graphs(output_graph, [g])
    print(f"✓ Graph saved to {output_graph}")
    print(f"    → Nodes: paper={g.num_nodes('paper'):,}, fos={g.num_nodes('fos'):,}, topic={g.num_nodes('topic'):,}")
    print(f"    → Edges: cites={g.num_edges('cites'):,}, has_fos={g.num_edges('has_fos'):,}, has_topic={g.num_edges('has_topic'):,}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build DGL knowledge graph without SciBERT embeddings")
    parser.add_argument("--csv", default="/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/topic_modeled_dblp.csv")
    parser.add_argument("--bertopic", default="/media/e-soc-student/DISK2/GR/GR2_Recommendation/models/bertopic_online.pkl")
    parser.add_argument("--out", default="/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/paper_graph_dgl.bin")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    build_heterograph(args.csv, args.bertopic, args.out, device=args.device)
