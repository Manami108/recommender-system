import pandas as pd

# Input and output file paths
INPUT_DOI_FILE = "/home/abhi/Desktop/Manami/recommender-system/datasets/filtered_2020_doi_title.csv"
OUTPUT_JSONL = "/home/abhi/Desktop/Manami/recommender-system/datasets/testset_2020.jsonl"

import json

# Load the DOIs
df = pd.read_csv(INPUT_DOI_FILE)
if "doi" not in df.columns:
    raise ValueError("Input file must contain a 'doi' column")

# Prepare section labels
sections = ["Introduction", "Related Work", "Methodology"]
assigned_sections = [sections[i % 3] for i in range(len(df))]

# Write JSONL
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
    for i, row in df.iterrows():
        item = {
            "paragraph": "",
            "section": assigned_sections[i],
            "doi": row["doi"]
        }
        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"JSONL file with 2000 paragraph frames saved to:\n{OUTPUT_JSONL}")
