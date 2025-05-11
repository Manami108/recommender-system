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
            nn.Linear(128, 1)  # Output: score for each action
        )

    def forward(self, relation_embeddings_batch):
        """
        relation_embeddings_batch: Tensor of shape (num_actions, embedding_dim)
        Returns:
            probs: Tensor of shape (num_actions,) with softmax probabilities
        """
        scores = self.policy_network(relation_embeddings_batch).squeeze(-1)  # shape: (num_actions,)
        probs = F.softmax(scores, dim=-1)
        return probs
