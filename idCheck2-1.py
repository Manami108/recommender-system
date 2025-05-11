import pandas as pd
import ast

# Load cleaned CSV with correct dtype
cleaned_csv = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/cleaned_dblp.csv"
df = pd.read_csv(cleaned_csv, dtype=str, low_memory=False)

# Build valid paper ID set
valid_ids = set(df['id'])

# Define stricter reference cleaner
def clean_valid_references(ref_string):
    try:
        refs = ast.literal_eval(ref_string) if isinstance(ref_string, str) else []
        if isinstance(refs, list):
            # Keep only IDs that are in valid_ids and not obviously invalid
            return str([ref for ref in refs if isinstance(ref, str) and ref in valid_ids])
    except Exception:
        pass
    return "[]"

# Apply reference cleaning again
print("Re-cleaning and validating references...")
df['references'] = df['references'].apply(clean_valid_references)

# Re-check for invalid references
invalid_ref_count = 0
invalid_references = []

for _, row in df.iterrows():
    try:
        refs = ast.literal_eval(row['references']) if isinstance(row['references'], str) else []
        for ref_id in refs:
            if ref_id not in valid_ids:
                invalid_references.append((row['id'], ref_id))
                invalid_ref_count += 1
    except Exception:
        continue

if invalid_ref_count == 0:
    print("All references are now valid.")
else:
    print(f"Still found {invalid_ref_count} invalid references.")
    print("First 10 invalid cases:", invalid_references[:10])

# Overwrite the cleaned CSV if all good
if invalid_ref_count == 0:
    df.to_csv(cleaned_csv, index=False)
    print("Cleaned dataset with valid references saved again.")
