import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

# === Paths ===
node_file = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/graph_nodes.csv"
edge_file = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/graph_edges.csv"
output_path = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/visualization/graph_visualization2_styled.png"

# === Load nodes and edges ===
print("📥 Loading graph data...")
nodes_df = pd.read_csv(node_file)
edges_df = pd.read_csv(edge_file)

# === Sample 10 random papers ===
print("🎲 Sampling 10 random papers...")
sampled_papers = nodes_df.sample(n=10, random_state=42)["paperId:ID"].tolist()

# === Filter edges involving sampled papers ===
print("🔗 Filtering edges...")
filtered_edges = edges_df[
    edges_df[":START_ID"].isin(sampled_papers) | edges_df[":END_ID"].isin(sampled_papers)
]

# === Build graph ===
print("🧠 Building NetworkX graph...")
G = nx.DiGraph()

# Add sampled nodes with attributes
for _, row in nodes_df[nodes_df["paperId:ID"].isin(sampled_papers)].iterrows():
    G.add_node(
        row["paperId:ID"],
        title=row.get("title", ""),
        topic=row.get("topic", -1),
        year=row.get("year", ""),
        n_citation=row.get("n_citation", 0)
    )

# Add filtered citation edges
for _, row in filtered_edges.iterrows():
    G.add_edge(row[":START_ID"], row[":END_ID"])

# === Prepare styles ===

# 1. Node color by topic (safe fallback if missing)
topic_values = nx.get_node_attributes(G, "topic")
unique_topics = sorted(set(topic_values.values()))
topic_to_color = {topic: i for i, topic in enumerate(unique_topics)}

node_colors = [
    cm.get_cmap("tab20")(
        topic_to_color.get(topic_values.get(node, -1), 0) / max(1, len(unique_topics))
    )
    for node in G.nodes()
]


# 2. Node size by citation count (clipped)
citation_counts = nx.get_node_attributes(G, "n_citation")
node_sizes = [
    min(300, 10 + int(citation_counts.get(n, 0))) for n in G.nodes()
]

# === Draw Graph ===
print("🎨 Drawing graph...")

plt.figure(figsize=(18, 14))
pos = nx.spring_layout(G, k=0.5, seed=42)

# Nodes
nx.draw_networkx_nodes(
    G, pos,
    node_color=node_colors,
    node_size=node_sizes,
    alpha=0.85,
    edgecolors="black",
    linewidths=0.5
)

# Edges
nx.draw_networkx_edges(
    G, pos,
    edge_color="gray",
    arrows=True,
    arrowstyle="-|>",
    arrowsize=12,
    connectionstyle="arc3,rad=0.15",
    alpha=0.5
)

# Optional: show node labels (small graphs only)
# labels = {n: G.nodes[n]["title"][:30] for n in G.nodes()}
# nx.draw_networkx_labels(G, pos, labels, font_size=7)

plt.title("📚 Knowledge Graph: Citation Network of 10 Random Papers", fontsize=16)
plt.axis("off")
plt.tight_layout()

# Save to file
plt.savefig(output_path, dpi=300)
print(f"✅ Graph saved to: {output_path}")
