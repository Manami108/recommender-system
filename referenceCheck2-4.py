import ast
import pandas as pd

# Load
df = pd.read_csv("/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/topic_modeled_dblp.csv")

# Fix references
df['references'] = df['references'].fillna("[]")

# Step 1: Drop rows where 'id' is missing
df = df.dropna(subset=['id'])

# Step 2: Correct ID parsing
df['id'] = df['id'].apply(lambda x: str(int(float(x))))  # safe: no NaN now

# Step 3: How many papers have at least one reference
count_with_references = df['references'].apply(lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else 0)
print(f"Number of papers with references: {(count_with_references > 0).sum()} / {len(df)}")

# Step 4: Collect valid paper IDs
valid_ids = set(df['id'])

# Step 5: Count valid references
def count_valid_references(refs):
    try:
        parsed = ast.literal_eval(refs)
        if isinstance(parsed, list):
            return sum(1 for r in parsed if str(int(float(r))) in valid_ids)
    except:
        return 0

df['valid_citation_targets'] = df['references'].apply(count_valid_references)
print(f"Total valid citation links found: {df['valid_citation_targets'].sum()}")
