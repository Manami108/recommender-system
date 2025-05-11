import ast
import re
import pandas as pd

# === Safe FOS evaluation ===
def safe_literal_eval(s):
    try:
        s = re.sub(r"Decimal\('([^']+)'\)", r"'\1'", s)  # Remove Decimal('...')
        return ast.literal_eval(s)
    except Exception:
        return None

# === Load Dataset ===
df = pd.read_csv("/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/topic_modeled_dblp.csv")
df['fos'] = df['fos'].fillna("[]")

# === Correct FOS parsing ===
print("Parsing FOS fields safely...")
df['fos_parsed'] = df['fos'].apply(safe_literal_eval)

# === How many papers have FOS ===
df['num_fos'] = df['fos_parsed'].apply(lambda x: len(x) if isinstance(x, list) else 0)

print(f"Number of papers with at least one FOS: {(df['num_fos'] > 0).sum()} / {len(df)}")
print(f"Average number of FOS per paper: {df['num_fos'].mean():.2f}")

# === Collect unique FOS labels ===
fos_counter = {}

for parsed in df['fos_parsed']:
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and 'name' in item:
                fos_name = item['name'].strip()
                fos_counter[fos_name] = fos_counter.get(fos_name, 0) + 1
            elif isinstance(item, str):
                fos_counter[item.strip()] = fos_counter.get(item.strip(), 0) + 1

print(f"Total unique FOS labels: {len(fos_counter)}")

# Optional: Show top 10
print("Top 10 common FOS labels:")
for i, (k, v) in enumerate(sorted(fos_counter.items(), key=lambda x: x[1], reverse=True)[:10]):
    print(f"{i+1}. {k}: {v}")
