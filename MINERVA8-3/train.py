# train.py
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from environment import KGEnvironment
from agent import MINERVAAgent

# =============================== Parameters ===============================
embedding_dim = 50
num_epochs = 1000
max_steps = 5
learning_rate = 1e-3
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
triples_path = "./datasets/minerva_triples.tsv"
save_model_path = "./minerva_agent.pth"

# =============================== Load Triples ===============================
print("Loading triples...")
if not os.path.exists(triples_path):
    raise FileNotFoundError(f"Triples file not found: {triples_path}")

triples = []
entities = set()
relations = set()

with open(triples_path, "r") as f:
    for line in f:
        head, relation, tail = line.strip().split("\t")
        triples.append((head, relation, tail))
        entities.update([head, tail])
        relations.add(relation)

entities = sorted(entities)
relations = sorted(relations)

entity2id = {e: i for i, e in enumerate(entities)}
relation2id = {r: i for i, r in enumerate(relations)}
id2relation = {i: r for r, i in relation2id.items()}

print(f"Total Entities: {len(entities)}, Total Relations: {len(relations)}, Total Triples: {len(triples)}")

# =============================== Initialize Environment and Agent ===============================
env = KGEnvironment(triples)
agent = MINERVAAgent(num_relations=len(relations), embedding_dim=embedding_dim).to(device)
optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

# =============================== Training Loop ===============================
print("Starting training...")

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

            if not possible_actions:
                break  # No outgoing edges

            # Prepare action relations
            available_relations = [relation2id[r] for r, _ in possible_actions]
            available_relation_tensor = torch.tensor(available_relations, device=device)

            relation_embeddings = agent.relation_embeddings(available_relation_tensor)
            probs = agent(relation_embeddings.unsqueeze(0)).squeeze(0)
            probs = probs / probs.sum()  # Normalize

            # Sample an action
            action_idx = torch.multinomial(probs, num_samples=1).item()
            chosen_relation, next_entity = possible_actions[action_idx]

            current_entity = next_entity

            # Check if reached goal
            if current_entity == goal_entity:
                reward = 1
                break

        # Calculate loss
        if reward > 0:
            loss = -torch.log(probs[action_idx] + 1e-8)  # encourage
        else:
            loss = torch.log(probs[action_idx] + 1e-8)    # discourage

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}/{num_epochs} | Avg Loss: {total_loss / batch_size:.4f}")

print("Training complete.")

# =============================== Save Agent ===============================
torch.save(agent.state_dict(), save_model_path)
print(f"Model saved to {save_model_path}")
