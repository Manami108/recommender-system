import pandas as pd
import ast
import re

# =============================== File Paths ===============================
preprocessed_csv = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/preprocessed_dblp_FIXED.csv"
topic_modeled_csv = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/topic_modeled_dblp.csv"

# =============================== Load Data ===============================
print("Loading datasets...")
df_preprocessed = pd.read_csv(preprocessed_csv)
df_topicmodeled = pd.read_csv(topic_modeled_csv)

# =============================== Paper Counts ===============================
num_papers = len(df_topicmodeled)
print(f"Total number of papers: {num_papers}")

# =============================== Unique Topic Counts ===============================
topics = df_topicmodeled['topic'].dropna().unique()
topics = [str(t).strip() for t in topics if t != -1]
num_unique_topics = len(topics)
print(f"Total unique topics: {num_unique_topics}")

# =============================== Unique FOS Counts ===============================
def safe_literal_eval(s):
    try:
        s = re.sub(r"Decimal\('([^']+)'\)", r"'\1'", s)  # Remove Decimal('...')
        return ast.literal_eval(s)
    except:
        return None

df_preprocessed['fos'] = df_preprocessed['fos'].fillna("[]")
df_preprocessed['fos_parsed'] = df_preprocessed['fos'].apply(safe_literal_eval)

unique_fos = set()
for parsed in df_preprocessed['fos_parsed']:
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and 'name' in item:
                unique_fos.add(item['name'].strip())
            elif isinstance(item, str):
                unique_fos.add(item.strip())
num_unique_fos = len(unique_fos)
print(f"Total unique FOS: {num_unique_fos}")

# =============================== Total Citations (Safe) ===============================
# First, safely convert n_citation to numeric (errors='coerce' will turn non-numeric into NaN)
df_preprocessed['n_citation'] = pd.to_numeric(df_preprocessed['n_citation'], errors='coerce').fillna(0)

total_citations = df_preprocessed['n_citation'].sum()
print(f"Total number of citations (sum of n_citation field): {total_citations}")

# =============================== Total Assigned FOS (including duplicates) ===============================
total_fos_assigned = 0
for parsed in df_preprocessed['fos_parsed']:
    if isinstance(parsed, list):
        total_fos_assigned += len(parsed)
print(f"Total number of assigned FOS labels (count with duplicates): {total_fos_assigned}")

# =============================== Total Assigned Topics (including duplicates) ===============================
total_topic_assigned = (df_topicmodeled['topic'] != -1).sum()
print(f"Total number of assigned topics (count with duplicates): {total_topic_assigned}")

# =============================== Final Summary ===============================
print("\n=== Summary ===")
print(f"Papers: {num_papers}")
print(f"Unique Topics: {num_unique_topics}")
print(f"Unique FOS: {num_unique_fos}")
print(f"Total Citations: {total_citations}")
print(f"Total Assigned FOS (duplicates counted): {total_fos_assigned}")
print(f"Total Assigned Topics (duplicates counted): {total_topic_assigned}")
