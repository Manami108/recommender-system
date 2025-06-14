import pandas as pd
import ast
import random

# ======= CONFIGURATION =======
CSV_PATH = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/big/dblp.v12.csv"
OUTPUT_CSV = "/home/abhi/Desktop/Manami/recommender-system/datasets/testset_300_doi_title.csv"
SAMPLE_SIZE = 300
# =============================

def is_valid_row(row):
    """Check for year 2020, non-empty title, doi, abstract, and at least 15 references."""
    try:
        # Year filter
        if row.get("year") != 2020:
            return False

        # Required fields
        if pd.isna(row["title"]) or str(row["title"]).strip() == "":
            return False
        if pd.isna(row["doi"]) or str(row["doi"]).strip() == "":
            return False
        if pd.isna(row["indexed_abstract"]) or str(row["indexed_abstract"]).strip() == "":
            return False

        # Handle 'references' column safely
        references_str = row.get("references", "[]")
        if not isinstance(references_str, str) or references_str.strip() in ["", "[]", "nan"]:
            return False

        references = ast.literal_eval(references_str)
        if not isinstance(references, list) or len(references) < 15:
            return False

        return True
    except Exception:
        return False

def main():
    print(f"Loading CSV from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"Original dataset size: {len(df):,}")

    print("Filtering valid 2020 papers...")
    filtered_df = df[df.apply(is_valid_row, axis=1)]
    print(f"Filtered valid entries from 2020: {len(filtered_df):,}")

    if len(filtered_df) < SAMPLE_SIZE:
        raise ValueError(f"Only {len(filtered_df)} valid 2020 papers found. Cannot sample {SAMPLE_SIZE}.")

    print(f"Sampling {SAMPLE_SIZE} entries...")
    sampled_df = filtered_df.sample(n=SAMPLE_SIZE, random_state=42)

    print(f"Saving to: {OUTPUT_CSV}")
    sampled_df[["doi", "title"]].to_csv(OUTPUT_CSV, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
