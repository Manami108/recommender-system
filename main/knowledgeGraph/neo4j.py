#!/usr/bin/env python

# Exports six CSVs for neo4j-admin import (Paper, Topic, FoS + edges)
# — Every Paper has a 768-d SciBERT embedding (title+abstract)
# — Every Topic has keywords (+ optional embedding)
# — Every FoS has a SciBERT embedding
# If a FoS name in the paper CSV is not found in small_fos_index.json,
# the script auto-adds a new FoS node so that HAS_FOS edges are never lost.
# Topic keywords are encoded using SciBERT

#!/usr/bin/env python
import os
import re
import ast
import csv
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# ─────────────────────────── CONFIG ─────────────────────────── #
CSV_PATH   = "/home/abhi/Desktop/Manami/topic_modeled_SciBERT_tuned2.csv"
ORIG_DB    = "/home/abhi/Desktop/Manami/dblp.v12.csv"
BERTOPIC   = "/home/abhi/Desktop/Manami/bertopic_online_SciBERT_tuned2.pkl"
PAPER_EMB  = "/home/abhi/Desktop/Manami/recommender-system/embeddings_sciBERT.npy"
FOS_EMB    = "/home/abhi/Desktop/Manami/fos_embeddings_sciBERT.npy"
FOS_JSON   = "/home/abhi/Desktop/Manami/fos_index.json"
OUTPUT_DIR = "/home/abhi/Desktop/Manami/export_csv"

EMBED_TOPICS    = True
N_TOPIC_WORDS   = 10
os.makedirs(OUTPUT_DIR, exist_ok=True)

_decimal_pat = re.compile(r"Decimal\('([^']+)'\)")

def safe_eval(s):
    if not isinstance(s, str):
        return []
    try:
        cleaned = _decimal_pat.sub(r"\1", s)
        obj = ast.literal_eval(cleaned)
        return obj if isinstance(obj, list) else []
    except Exception:
        return []

def reconstruct(idx_str):
    try:
        obj = ast.literal_eval(idx_str)
        toks = [""] * obj.get("IndexLength", 0)
        for w, positions in obj.get("InvertedIndex", {}).items():
            for p in positions:
                if 0 <= p < len(toks):
                    toks[p] = w
        return " ".join(toks).strip()
    except Exception:
        return ""

vec_to_str = lambda v: ";".join(f"{x:.6f}" for x in v)

# ────────────────────────── LOAD DATA ────────────────────────── #
print("Loading original DBLP…")
orig = pd.read_csv(ORIG_DB, dtype={"id": str}, low_memory=False)

print("Loading topic assignments…")
topics = (
    pd.read_csv(CSV_PATH, usecols=["id","topic"], dtype={"id": str})
      .dropna(subset=["id"])
      .drop_duplicates("id", keep="first")
)

print("Merging…")
df = orig.merge(topics, on="id", how="left")
df["topic"] = df["topic"].fillna(-1).astype(int)

# fill NaNs
for col, fill in [
    ("references","[]"),
    ("fos","[]"),
    ("title",""),
    ("indexed_abstract",""),
    ("authors",""),
    ("year", -1),
]:
    df[col].fillna(fill, inplace=True)

# sanitize newlines
newline_re = re.compile(r"[\r\n]+")
for col in df.select_dtypes(include="object"):
    df[col] = df[col].str.replace(newline_re, " ", regex=True)

# embeddings
print("Loading embeddings…")
paper_emb = np.load(PAPER_EMB).astype(np.float32)
fos_emb   = np.load(FOS_EMB).astype(np.float32)

# BERTopic
print("Loading BERTopic…")
topic_model = BERTopic.load(BERTOPIC)
topic_ids   = topic_model.get_topic_info().query("Topic != -1").Topic.astype(int).tolist()
topic_idx   = {tid: i for i, tid in enumerate(topic_ids)}

if EMBED_TOPICS:
    print("Encoding topic keywords…")
    st = SentenceTransformer("allenai/scibert_scivocab_uncased")
    topic_emb = np.vstack([
        st.encode(" ".join(w for w,_ in topic_model.get_topic(tid)[:N_TOPIC_WORDS]),
                  convert_to_numpy=True)
        for tid in topic_ids
    ]).astype(np.float32)

# FoS index
with open(FOS_JSON) as f:
    fos_name2idx = {k.lower(): v for k, v in json.load(f).items()}
next_fos_idx = max(fos_name2idx.values(), default=-1) + 1

# ────────────────────────── PREPARE CSVs ────────────────────────── #
open_kw = dict(mode="w", newline="", encoding="utf-8")
def writer(path):
    return csv.writer(open(path, **open_kw), quoting=csv.QUOTE_MINIMAL)

paper_w = writer(f"{OUTPUT_DIR}/paper_nodes.csv")
topic_w = writer(f"{OUTPUT_DIR}/topic_nodes.csv")
fos_w   = writer(f"{OUTPUT_DIR}/fos_nodes.csv")
cites_w = writer(f"{OUTPUT_DIR}/paper_cites_paper.csv")
ht_w    = writer(f"{OUTPUT_DIR}/paper_has_topic.csv")
hf_w    = writer(f"{OUTPUT_DIR}/paper_has_fos.csv")

# headers
paper_w.writerow([
    "pid:ID(Paper)",
    "doi",
    "title",
    "abstract",
    "authors",
    "year:int",
    "embedding:float[]"
])
topic_w.writerow(
    ["idx:ID(Topic)", "topic_id", "keywords"] +
    (["embedding:float[]"] if EMBED_TOPICS else [])
)
fos_w.writerow(["idx:ID(FieldOfStudy)", "name", "embedding:float[]"])
cites_w.writerow([":START_ID(Paper)", ":END_ID(Paper)"])
ht_w.writerow([":START_ID(Paper)", ":END_ID(Topic)"])
hf_w.writerow([":START_ID(Paper)", ":END_ID(FieldOfStudy)"])

# write topic nodes
for i, tid in enumerate(topic_ids):
    row = [i, tid, " ".join(w for w,_ in topic_model.get_topic(tid)[:N_TOPIC_WORDS])]
    if EMBED_TOPICS:
        row.append(vec_to_str(topic_emb[i]))
    topic_w.writerow(row)

# write FoS nodes
for name, idx in fos_name2idx.items():
    fos_w.writerow([idx, name, vec_to_str(fos_emb[idx])])

# ────────────────────────── EXPORT NODES & EDGES ────────────────────────── #
print("Exporting papers & edges…")
valid_ids = set(df["id"])
ecites = ehf = eht = 0

for i, row in tqdm(df.iterrows(), total=len(df)):
    pid  = row["id"]
    doi  = row.get("doi","") or ""
    title= row["title"]
    abst = reconstruct(row["indexed_abstract"])
    auth = row["authors"]
    year = int(row["year"]) if str(row["year"]).isdigit() else -1
    emb  = vec_to_str(paper_emb[i])

    # paper node
    paper_w.writerow([pid, doi, title, abst, auth, year, emb])

    # CITES edges
    for ref in safe_eval(row["references"]):
        rid = str(ref).replace(".0","").strip()
        if rid in valid_ids:
            cites_w.writerow([pid, rid])
            ecites += 1

    # HAS_FOS edges
    for f in safe_eval(row["fos"]):
        name = (f["name"] if isinstance(f, dict) else f).strip().lower()
        if not name:
            continue
        if name not in fos_name2idx:
            fos_name2idx[name] = next_fos_idx
            fos_w.writerow([next_fos_idx, name, vec_to_str(np.zeros(768, dtype=np.float32))])
            next_fos_idx += 1
        hf_w.writerow([pid, fos_name2idx[name]])
        ehf += 1

    # HAS_TOPIC edge
    tid = row["topic"]
    if tid != -1 and tid in topic_idx:
        ht_w.writerow([pid, topic_idx[tid]])
        eht += 1

print("Done.")
print(f"Papers:   {len(df):,}")
print(f"CITES:    {ecites:,}")
print(f"HAS_FOS:  {ehf:,}")
print(f"HAS_TOPIC:{eht:,}")
