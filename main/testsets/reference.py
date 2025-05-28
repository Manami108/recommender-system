import json
import pandas as pd

# ====== File Paths ======
INPUT_JSONL = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/big/testset_300.jsonl"  # this is JSONL
DBLP_CSV = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/big/dblp.v12.csv"
OUTPUT_JSONL = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/big/testset_300_references.jsonl"
# ========================

# Step 1: Load JSONL entries
entries = []
with open(INPUT_JSONL, "r", encoding="utf-8") as f_in:
    for _ in range(100):
        line = f_in.readline()
        if not line:
            break
        item = json.loads(line)
        # Drop if paragraph is just "1", "2", or "3"
        if str(item.get("paragraph", "")).strip() in {"1", "2", "3"}:
            continue
        entries.append(item)

print(f"Valid entries after filtering: {len(entries)}")

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
            ref_list = json.loads(refs.replace("'", '"'))  # Ensure proper list parsing
            if isinstance(ref_list, list):
                doi_to_refs[doi] = ref_list
    except:
        continue  # Skip broken entries

# Step 3: Add references to JSON entries
for item in entries:
    doi = str(item.get("doi", "")).strip()
    item["references"] = doi_to_refs.get(doi, [])

# Step 4: Save back to JSONL
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
    for item in entries:
        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Saved final JSONL with references to:\n{OUTPUT_JSONL}")
