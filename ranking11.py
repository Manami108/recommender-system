import pandas as pd
import numpy as np

# === Assumptions ===
# You already have:
# - List of Top-K candidate paper dicts from previous step
# - Each candidate has: paper_id, similarity_score, topic_match, writing_phase
# - Your paragraph's writing phase is known (from LLM)

# Load paper metadata
df = pd.read_csv("/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/topic_modeled_dblp.csv")

# === Scoring Weights ===
alpha = 0.5   # embedding similarity
beta = 0.0    # citation_graph_score (not used yet)
gamma = 0.3   # topic alignment
delta = 0.0   # fos diversity (optional later)
theta = 0.2   # phase match bonus

# === Phase-specific logic ===
def phase_match_boost(phase, paper_row):
    """
    Assign a bonus score depending on citation intent of phase.
    Example:
    - Intro → high citation count
    - Related Work → method/topic overlap
    """
    if phase == "Introduction":
        return float(paper_row.get("n_citation", 0)) / 1000.0  # normalize
    elif phase == "Related Work":
        return 1.0  # we’ll count this fully
    elif phase == "Method":
        return 0.5  # partial relevance
    else:
        return 0.2

# === Main Re-ranking Function ===
def rerank_candidates(candidates, writing_phase, target_topic):
    reranked = []
    for c in candidates:
        paper_row = df[df["id"] == c["paper_id"]]
        if paper_row.empty:
            continue
        paper = paper_row.iloc[0]

        # Features
        sim = c["similarity_score"]
        topic_score = 1.0 if int(paper["topic"]) == target_topic else 0.0
        phase_bonus = phase_match_boost(writing_phase, paper)

        # Final score
        final_score = alpha * sim + gamma * topic_score + theta * phase_bonus
        explanation = []
        explanation.append(f"✅ Similar to your paragraph (sim={sim:.2f})")
        if topic_score:
            explanation.append(f"🧠 Shares topic: {target_topic}")
        if phase_bonus > 0.5:
            explanation.append(f"📚 Good fit for phase: {writing_phase}")

        reranked.append({
            "paper_id": c["paper_id"],
            "title": c["title"],
            "score": round(final_score, 4),
            "explanation": "; ".join(explanation)
        })

    # Sort by final score
    return sorted(reranked, key=lambda x: x["score"], reverse=True)
