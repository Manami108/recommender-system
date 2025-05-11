import pandas as pd
import ast
import numpy as np
import re

# Helper function to remove Decimal wrappers before literal_eval
def safe_literal_eval(s):
    try:
        # Replace patterns like Decimal('0.51544') with just '0.51544'
        s = re.sub(r"Decimal\('([^']+)'\)", r"'\1'", s)
        return ast.literal_eval(s)
    except Exception as e:
        # You can also print(e) for debugging if needed
        return None

# Load your preprocessed CSV (assumed to contain a "fos" column)
preprocessed_csv = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/preprocessed_dblp.csv"
df = pd.read_csv(preprocessed_csv, low_memory=False)

# Build the set of unique fos labels
unique_fos = set()

for fos_str in df['fos'].dropna():
    parsed = safe_literal_eval(fos_str)
    if parsed and isinstance(parsed, list):
        for item in parsed:
            # If the item is a dict with a 'name' key, extract its value.
            if isinstance(item, dict) and 'name' in item:
                fos_label = item['name']
            elif isinstance(item, str):
                fos_label = item
            else:
                continue

            unique_fos.add(fos_label.lower().strip())

unique_fos = sorted(list(unique_fos))
fos_to_idx = {fos: i for i, fos in enumerate(unique_fos)}
num_fos = len(unique_fos)

print(f"Found {num_fos} unique fields of study (fos).")

def get_fos_vector(fos_str, fos_to_idx, vector_size):
    """Convert the fos string (a list in string form) into a multi-hot vector."""
    vector = np.zeros(vector_size, dtype=np.float32)
    parsed = safe_literal_eval(fos_str)
    if parsed and isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and 'name' in item:
                key = item['name']
            elif isinstance(item, str):
                key = item
            else:
                continue
            key = key.lower().strip()
            if key in fos_to_idx:
                vector[fos_to_idx[key]] = 1.0
    return vector

# Example usage: print the multi-hot vector for the first row's fos (if available)
if not df['fos'].dropna().empty:
    example_fos = df['fos'].dropna().iloc[0]
    vector = get_fos_vector(example_fos, fos_to_idx, num_fos)
    print("Example multi-hot vector for fos:", vector)
