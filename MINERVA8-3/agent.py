# agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MINERVAAgent(nn.Module):
    def __init__(self, num_relations, embedding_dim):
        super(MINERVAAgent, self).__init__()
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.policy_network = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Single score for each relation
        )

    def forward(self, relation_embeddings_batch):
        logits = self.policy_network(relation_embeddings_batch)  # shape: (batch_size, 1)
        logits = logits.squeeze(-1)  # (batch_size,)
        probs = F.softmax(logits, dim=-1)
        return probs
