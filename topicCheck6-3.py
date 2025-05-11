#!/usr/bin/env python
# ──────────────  Print Top 100 Topics ────────────────────────────────
from bertopic import BERTopic
import os

# ─── Path to the Saved Model ─────────────────────────────────────────
MODEL_PATH = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/models/bertopic_online.pkl"

# ─── Load the Model ──────────────────────────────────────────────────
print(f"▶ Loading BERTopic model from: {MODEL_PATH}")
topic_model = BERTopic.load(MODEL_PATH)

# ─── Extract and Print Top 100 Topics ────────────────────────────────
print("\n▶ Top 100 Topics by Frequency:")
top_100 = topic_model.get_topic_freq().head(100)

for i, row in top_100.iterrows():
    topic_id = row["Topic"]
    count = row["Count"]
    if topic_id == -1:
        print(f"Topic {topic_id} [outliers] — {count} docs")
    else:
        words = topic_model.get_topic(topic_id)
        keywords = ", ".join([word for word, _ in words[:10]])
        print(f"Topic {topic_id} — {count} docs — {keywords}")

print("\n✓ Done.")
