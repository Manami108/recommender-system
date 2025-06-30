import json
import pandas as pd
import random

# file path
INPUT_JSONL = "/home/abhi/Desktop/Manami/recommender-system/datasets/testset_2020.jsonl"
DBLP_CSV = "/home/abhi/Desktop/Manami/recommender-system/datasets/dblp.v12.csv"
OUTPUT_JSONL_ALL = "/home/abhi/Desktop/Manami/recommender-system/datasets/testset_2020_references.jsonl"
OUTPUT_JSONL_1 = "/home/abhi/Desktop/Manami/recommender-system/datasets/testset1.jsonl"
OUTPUT_JSONL_2 = "/home/abhi/Desktop/Manami/recommender-system/datasets/testset2.jsonl"

# Step 1: Load JSONL entries
entries = []
with open(INPUT_JSONL, "r", encoding="utf-8") as f_in:
    for _ in range(300):  # read enough lines
        line = f_in.readline()
        if not line:
            break
        item = json.loads(line)
        # Drop if paragraph is just "1", "2", or "3"
        para = str(item.get("paragraph", "")).strip()
        if para in {"1", "2", "3"}:
            continue
        # Check paragraph length > 300 words
        word_count = len(para.split())
        if word_count <= 400:
            continue
        entries.append(item)

print(f"Valid entries after filtering (>{400} words): {len(entries)}")

# Ensure we have at least 200 entries
if len(entries) < 200:
    raise ValueError(f"Not enough entries with >300 words: found {len(entries)}")

# Step 2: Load DBLP dataset
print("Loading DBLP dataset...")
dblp_df = pd.read_csv(DBLP_CSV, usecols=["doi", "references"], low_memory=False)
dblp_df["doi"] = dblp_df["doi"].astype(str).str.strip()

# Convert to lookup dictionary
doi_to_refs = {}
for _, row in dblp_df.iterrows():
    doi = str(row["doi"]).strip()
    refs = row["references"]
    try:
        if isinstance(refs, str):
            ref_list = json.loads(refs.replace("'", '"'))
            if isinstance(ref_list, list):
                doi_to_refs[doi] = ref_list
    except json.JSONDecodeError:
        continue

# Step 3: Add references to all filtered entries
for item in entries:
    doi = str(item.get("doi", "")).strip()
    item["references"] = doi_to_refs.get(doi, [])

# Save the full filtered set with references
with open(OUTPUT_JSONL_ALL, "w", encoding="utf-8") as f_out:
    for item in entries:
        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved full filtered JSONL with references to: {OUTPUT_JSONL_ALL}")

# Step 4: Split into two random testsets of 100 each
random.seed(42)  # for reproducibility
random.shuffle(entries)

testset1 = entries[:100]
testset2 = entries[100:200]

# Save testset1
with open(OUTPUT_JSONL_1, "w", encoding="utf-8") as f1:
    for item in testset1:
        f1.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved testset1 (100 entries) to: {OUTPUT_JSONL_1}")

# Save testset2
with open(OUTPUT_JSONL_2, "w", encoding="utf-8") as f2:
    for item in testset2:
        f2.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved testset2 (100 entries) to: {OUTPUT_JSONL_2}")
