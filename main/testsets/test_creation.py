import pandas as pd
import ast

# ======= CONFIGURATION =======
CSV_PATH    = "/home/abhi/Desktop/Manami/recommender-system/datasets/dblp.v12.csv"
OUTPUT_CSV  = "/home/abhi/Desktop/Manami/recommender-system/datasets/filtered_2020_doi_title.csv"
# =============================

def is_valid_row(row):
    """Check for year 2020, non-empty title, doi, abstract, and at least 15 references."""
    try:
        # Year filter
        if row.get("year") != 2020:
            return False

        # Required fields
        if pd.isna(row["title"]) or not str(row["title"]).strip():
            return False
        if pd.isna(row["doi"])   or not str(row["doi"]).strip():
            return False
        if pd.isna(row["indexed_abstract"]) or not str(row["indexed_abstract"]).strip():
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
    total_valid = len(filtered_df)
    print(f"Total valid entries from 2020: {total_valid:,}")

    if total_valid == 0:
        print("No papers found matching the criteria.")
        return

    print(f"Saving all {total_valid:,} entries to: {OUTPUT_CSV}")
    filtered_df[["doi", "title"]].to_csv(OUTPUT_CSV, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
