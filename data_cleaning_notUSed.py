import pandas as pd
import ast

# File paths
input_csv = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/dblp.v12.csv"
cleaned_csv = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/cleaned_dblp.csv"

print("Loading the dataset...")
df = pd.read_csv(input_csv)

# Ensure required columns exist
required_columns = ['id', 'title', 'indexed_abstract']
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Clean ID format
df['id'] = df['id'].astype(str).str.strip()

# Step 1: Remove rows where id OR title OR abstract is missing
df.replace({"": None}, inplace=True)  # Normalize empty strings to None
df_cleaned = df.dropna(subset=required_columns)  # Drop if any essential field is missing

# Step 2: Drop duplicates by ID and Title
df_cleaned = df_cleaned.drop_duplicates(subset=['id', 'title'])

# Step 3: Clean references (remove citations to dropped papers)
if 'references' in df_cleaned.columns:
    print("Cleaning references...")

    # Build set of valid paper IDs
    valid_ids = set(df_cleaned['id'])

    def clean_references(ref_string):
        if isinstance(ref_string, str):
            try:
                refs = ast.literal_eval(ref_string)
                if isinstance(refs, list):
                    # Keep only references pointing to existing IDs
                    return str([ref for ref in refs if ref in valid_ids])
            except (ValueError, SyntaxError):
                return "[]"
        return "[]"

    df_cleaned['references'] = df_cleaned['references'].apply(clean_references)

# Save the cleaned result
df_cleaned.to_csv(cleaned_csv, index=False)
print(f"Cleaned dataset saved to: {cleaned_csv}")
print(f"Total valid papers: {df_cleaned.shape[0]}")
