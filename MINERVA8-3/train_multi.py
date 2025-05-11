import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from environment import KGEnvironment
from agent_multi import MINERVAAgent
import hnswlib

# =============================== Parameters ===============================
embedding_dim = 50
num_epochs = 1000
max_steps = 5
learning_rate = 1e-3
batch_size = 8
top_k_text_neighbors = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
triples_path = "./datasets/minerva_triples.tsv"
scibert_embeddings_path = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/embeddings_sciBERT.npy"
hnsw_index_path = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/hnsw_index.bin"
df_path = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/preprocessed_dblp.csv"
save_model_path = "./minerva_agent_multilevel.pth"

# =============================== Load Triples ===============================
print("Loading triples...")
triples = []
entities = set()
relations = set()

with open(triples_path, "r") as f:
    for line in f:
        head, relation, tail = line.strip().split("\t")
        triples.append((head, relation, tail))
        entities.update([head, tail])
        relations.add(relation)

# =============================== Load Entity Order ===============================
print("Loading paper ID list used in embeddings...")
df = pd.read_csv(df_path, low_memory=False)
df = df.dropna(subset=['id'])
df['id'] = df['id'].apply(lambda x: str(int(float(x))))
paper_id_list = df['id'].tolist()
paper_id_to_index = {pid: i for i, pid in enumerate(paper_id_list)}

# Keep only entities that exist in the embedding space
entities = sorted(e for e in entities if e in paper_id_to_index)
entity2id = {e: i for i, e in enumerate(entities)}
id2entity = {i: e for e, i in entity2id.items()}
relation2id = {r: i for i, r in enumerate(sorted(relations))}
id2relation = {i: r for r, i in relation2id.items()}

print(f"Total Entities: {len(entities)}, Total Relations: {len(relations)}, Total Triples: {len(triples)}")

# =============================== Load Embeddings with Memory Mapping ===============================
print("Loading SciBERT embeddings (memory-mapped)...")
all_embeddings = np.load(scibert_embeddings_path, mmap_mode="r")  # ✅ mmap avoids RAM crash

# =============================== Load HNSW Index ===============================
print("Loading HNSW index...")
index = hnswlib.Index(space='cosine', dim=all_embeddings.shape[1])
index.load_index(hnsw_index_path)
print("HNSW index loaded.")

# =============================== Initialize Environment and Agent ===============================
env = KGEnvironment(triples)
agent = MINERVAAgent(num_relations=len(relations) + 1, embedding_dim=embedding_dim).to(device)
optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

# =============================== Training Loop ===============================
print("Starting multi-level training...")

for epoch in range(1, num_epochs + 1):
    agent.train()
    total_loss = 0.0
    batch = random.choices(triples, k=batch_size)

    for head, relation, tail in batch:
        current_entity = head
        goal_entity = tail
        target_relation = relation

        reward = 0
        loss = 0

        for step in range(max_steps):
            possible_actions = env.get_possible_actions(current_entity)
            graph_actions = [(relation2id[r], e) for r, e in possible_actions if r in relation2id]

            # === Semantic neighbors (from HNSW) ===
            text_actions = []
            if current_entity in paper_id_to_index:
                cur_idx = paper_id_to_index[current_entity]
                if cur_idx < len(all_embeddings): 
                    cur_emb = all_embeddings[cur_idx]
                    neighbor_indices, _ = index.knn_query(cur_emb, k=top_k_text_neighbors)
                    for nn_idx in neighbor_indices[0]:
                        if nn_idx < len(paper_id_list):
                            neighbor_id = paper_id_list[nn_idx]
                            if neighbor_id != current_entity:
                                text_actions.append((len(relation2id), neighbor_id))


            all_actions = graph_actions + text_actions
            if not all_actions:
                break

            relation_ids = torch.tensor([rel for rel, _ in all_actions], device=device)
            relation_embeds = agent.relation_embeddings(relation_ids)
            probs = agent(relation_embeds.unsqueeze(0)).squeeze(0)
            probs = probs / probs.sum()

            action_idx = torch.multinomial(probs, num_samples=1).item()
            chosen_relation_id, next_entity = all_actions[action_idx]
            current_entity = next_entity

            if current_entity == goal_entity:
                reward = 1
                break

        loss = -torch.log(probs[action_idx] + 1e-8) if reward > 0 else torch.log(probs[action_idx] + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch}/{num_epochs} | Avg Loss: {total_loss / batch_size:.4f}")

print("Training complete.")

# =============================== Save Model ===============================
torch.save(agent.state_dict(), save_model_path)
print(f"Model saved to {save_model_path}")
