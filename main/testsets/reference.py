import json
import pandas as pd
import random

# file path
INPUT_JSONL      = "/home/abhi/Desktop/Manami/recommender-system/datasets/testset_2020.jsonl"
DBLP_CSV         = "/home/abhi/Desktop/Manami/recommender-system/datasets/dblp.v12.csv"
OUTPUT_JSONL_ALL = "/home/abhi/Desktop/Manami/recommender-system/datasets/testset_2020_references.jsonl"
# new testset files
OUTPUT_JSONL_1   = "/home/abhi/Desktop/Manami/recommender-system/datasets/testset1.jsonl"
OUTPUT_JSONL_2   = "/home/abhi/Desktop/Manami/recommender-system/datasets/testset2.jsonl"
OUTPUT_JSONL_3   = "/home/abhi/Desktop/Manami/recommender-system/datasets/testset3.jsonl"
OUTPUT_JSONL_4   = "/home/abhi/Desktop/Manami/recommender-system/datasets/testset4.jsonl"

# Step 1: Load and filter JSONL entries
entries = []
with open(INPUT_JSONL, "r", encoding="utf-8") as f_in:
    for _ in range(300):
        line = f_in.readline()
        if not line:
            break
        item = json.loads(line)
        para = str(item.get("paragraph", "")).strip()
        if para in {"1", "2", "3"}:
            continue
        if len(para.split()) <= 400:
            continue
        entries.append(item)

print(f"Valid entries after filtering (>400 words): {len(entries)}")
if len(entries) < 200:
    raise ValueError(f"Not enough entries: found {len(entries)}")

# Step 2: Load DBLP and build DOI→references map
print("Loading DBLP dataset...")
dblp_df = pd.read_csv(DBLP_CSV, usecols=["doi", "references"], low_memory=False)
dblp_df["doi"] = dblp_df["doi"].astype(str).str.strip()

doi_to_refs = {}
for _, row in dblp_df.iterrows():
    doi = row["doi"]
    refs = row["references"]
    if isinstance(refs, str):
        try:
            ref_list = json.loads(refs.replace("'", '"'))
            if isinstance(ref_list, list):
                doi_to_refs[doi] = ref_list
        except json.JSONDecodeError:
            pass

# Step 3: Attach references to each entry
for item in entries:
    doi = str(item.get("doi", "")).strip()
    item["references"] = doi_to_refs.get(doi, [])

# Save the full filtered set with references
with open(OUTPUT_JSONL_ALL, "w", encoding="utf-8") as f_out:
    for item in entries:
        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved full filtered JSONL with references to: {OUTPUT_JSONL_ALL}")

# Step 4: Shuffle and split into 4 testsets of 50 each
random.seed(42)
random.shuffle(entries)

# define chunk size and slices
chunk_size = 50
testset1 = entries[0:chunk_size]
testset2 = entries[chunk_size:chunk_size*2]
testset3 = entries[chunk_size*2:chunk_size*3]
testset4 = entries[chunk_size*3:chunk_size*4]

# Save each testset
for idx, (subset, path) in enumerate([
    (testset1, OUTPUT_JSONL_1),
    (testset2, OUTPUT_JSONL_2),
    (testset3, OUTPUT_JSONL_3),
    (testset4, OUTPUT_JSONL_4),
]):
    with open(path, "w", encoding="utf-8") as f:
        for item in subset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved testset{idx+1} ({len(subset)} entries) to: {path}")
