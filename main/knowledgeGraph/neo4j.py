#!/usr/bin/env python

# Exports six CSVs for neo4j-admin import (Paper, Topic, FoS + edges)
# — Every Paper has a 768-d SciBERT embedding (title+abstract)
# — Every Topic has keywords (+ optional embedding)
# — Every FoS has a SciBERT embedding
# If a FoS name in the paper CSV is not found in small_fos_index.json,
# the script auto-adds a new FoS node so that HAS_FOS edges are never lost.
# Topic keywords are encoded using SciBERT

import os, re, ast, csv, json, numpy as np, pandas as pd
from tqdm import tqdm
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

CSV_PATH  = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/big/topic_modeled_SciBERT_tuned2.csv"
BERTOPIC  = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/models/bertopic_online_SciBERT_tuned2.pkl"
PAPER_EMB = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/big/embeddings_sciBERT.npy"
FOS_EMB   = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/big/fos_embeddings_sciBERT.npy"
FOS_JSON  = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/big/fos_index.json"
OUTPUT_DIR = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/export_csv"
EMBED_TOPICS, N_TOPIC_WORDS = True, 10
os.makedirs(OUTPUT_DIR, exist_ok=True)

_decimal_pat = re.compile(r"Decimal\('([^']+)'\)")

def safe_eval(s):
    if not isinstance(s, str):
        return []
    try:
        cleaned = _decimal_pat.sub(r"\1", s)
        obj = ast.literal_eval(cleaned)
        # only lists are valid here
        return obj if isinstance(obj, list) else []
    except Exception:
        return []

    
    
def reconstruct(idx_str):
    try:
        obj = ast.literal_eval(idx_str)
        toks = [""]*obj.get("IndexLength",0)
        for w,pos in obj.get("InvertedIndex",{}).items():
            for p in pos:
                if 0<=p<len(toks): toks[p]=w
        return " ".join(toks).strip()
    except Exception:
        return ""

vec_to_str = lambda v: ";".join(f"{x:.6f}" for x in v)

ORIG_DB = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/big/dblp.v12.csv"
MODELED = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/big/topic_modeled_SciBERT_tuned2.csv"

print("loading original DBLP CSV…")
orig = pd.read_csv(ORIG_DB, dtype={"id": str}, low_memory=False)

print("loading modeled topics…")
topics = (
    pd.read_csv(MODELED, usecols=["id","topic"], dtype={"id": str})
      .dropna(subset=["id"])
      .drop_duplicates(subset="id", keep="first")
)

print("▶ merging topics into original…")
df = orig.merge(topics, on="id", how="left")
df["topic"] = df["topic"].fillna(-1).astype(int)

# row count after merging
if len(df) != len(orig):
    raise ValueError(
        f"Merge row‐count mismatch:\n"
        f"  original rows: {len(orig):,}\n"
        f"  merged   rows: {len(df):,}\n"
        f"These must be equal before proceeding."
    )
print(f"Merge OK: {len(df):,} rows (matches original)")


# fill NaNs in other columns exactly as before
for col, fill in [("references","[]"),("fos","[]"),("title",""),
                  ("indexed_abstract",""),("authors",""),("year",-1)]:
    df[col].fillna(fill, inplace=True)

# sanitize any embedded newlines now that df is finalized
newline_re = re.compile(r"[\r\n]+")
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.replace(newline_re, " ", regex=True)


print("loading embeddings")
paper_emb = np.load(PAPER_EMB).astype(np.float32)
fos_emb   = np.load(FOS_EMB).astype(np.float32)

print("loading BERTopic")
topic_model = BERTopic.load(BERTOPIC)
topic_ids   = topic_model.get_topic_info().query("Topic != -1").Topic.astype(int).tolist()
topic_idx   = {tid:i for i,tid in enumerate(topic_ids)} 

if EMBED_TOPICS:
    print("encoding topic keywords")
    st = SentenceTransformer("allenai/scibert_scivocab_uncased")
    topic_emb = np.vstack([
        st.encode(" ".join(w for w,_ in topic_model.get_topic(tid)[:N_TOPIC_WORDS]),
                  convert_to_numpy=True)
        for tid in topic_ids ]).astype(np.float32)

# FoS mapping (name → idx) 
with open(FOS_JSON) as f:
    fos_name2idx = {k.lower():v for k,v in json.load(f).items()}
next_fos_idx = max(fos_name2idx.values()) + 1 if fos_name2idx else 0

# CSV writers 
open_kw = dict(mode="w", newline="", encoding="utf-8")
def writer(path): return csv.writer(open(path, **open_kw), quoting=csv.QUOTE_MINIMAL)

paper_w, topic_w  = writer(f"{OUTPUT_DIR}/paper_nodes.csv"),  writer(f"{OUTPUT_DIR}/topic_nodes.csv")
fos_w             = writer(f"{OUTPUT_DIR}/fos_nodes.csv")
cites_w, ht_w, hf_w = (writer(f"{OUTPUT_DIR}/{n}.csv") for n in
                      ["paper_cites_paper","paper_has_topic","paper_has_fos"])

paper_w.writerow(["id:ID(Paper)","title","abstract","authors","year:int","embedding:float[]"])
hdr = ["idx:ID(Topic)","topic_id","keywords"] + (["embedding:float[]"] if EMBED_TOPICS else [])
topic_w.writerow(hdr)
fos_w.writerow(["idx:ID(FieldOfStudy)","name","embedding:float[]"])
cites_w.writerow([":START_ID(Paper)",":END_ID(Paper)"])
ht_w.writerow([":START_ID(Paper)",":END_ID(Topic)"])
hf_w.writerow([":START_ID(Paper)",":END_ID(FieldOfStudy)"])

# write Topic + initial FoS nodes 
for i, tid in enumerate(topic_ids):
    row = [i, tid, " ".join(w for w,_ in topic_model.get_topic(tid)[:N_TOPIC_WORDS])]
    if EMBED_TOPICS: row.append(vec_to_str(topic_emb[i]))
    topic_w.writerow(row)

for name, idx in fos_name2idx.items():
    fos_w.writerow([idx, name, vec_to_str(fos_emb[idx])])
# ─── sanity check: row count consistency ─────────────────────────────
# if len(df) != paper_emb.shape[0]:
#     raise ValueError(
#         f"Row mismatch:\n"
#         f"  CSV rows:       {len(df):,}\n"
#         f"  Embedding rows: {paper_emb.shape[0]:,}\n"
#         f"These must match 1:1 before export."
#     )

# check for duplicate IDs
dup_ids = df["id"][df["id"].duplicated()].unique()
if len(dup_ids) > 0:
    print(f"Warning: {len(dup_ids)} duplicate Paper IDs found. Sample: {dup_ids[:5]}")

# papers & edges
print("exporting papers & edges")
valid_ids = set(df["id"])
ecites = ehf = eht = 0

for i, row in tqdm(df.iterrows(), total=len(df)):
    pid = row["id"]
    paper_w.writerow([
        pid, row["title"], reconstruct(row["indexed_abstract"]),
        row["authors"], int(row["year"]) if str(row["year"]).isdigit() else -1,
        vec_to_str(paper_emb[i])
    ])

    # CITES
    for ref in safe_eval(row["references"]):
        rid = str(ref).replace(".0","").strip()
        if rid in valid_ids:
            cites_w.writerow([pid, rid]); ecites+=1

    # HAS_FOS
    for f in safe_eval(row["fos"]):
        name = (f["name"] if isinstance(f,dict) else f).strip().lower()
        if not name: continue
        if name not in fos_name2idx:
            # unseen FoS → add node & embedding (zero vector placeholder)
            fos_name2idx[name] = next_fos_idx
            fos_w.writerow([next_fos_idx, name,
                            vec_to_str(np.zeros(768, dtype=np.float32))])
            next_fos_idx += 1
        hf_w.writerow([pid, fos_name2idx[name]]); ehf+=1
        
    tid = row["topic"]   
    if tid != -1 and tid in topic_idx:
        ht_w.writerow([pid, topic_idx[tid]])
        eht += 1


print("CSV export complete ", OUTPUT_DIR)
print(f"Papers {len(df):,} | Topics {len(topic_ids):,} | FoS {len(fos_name2idx):,}")
print(f"CITES {ecites:,} | HAS_TOPIC {eht:,} | HAS_FOS {ehf:,}")