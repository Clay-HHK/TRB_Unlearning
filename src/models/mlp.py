import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) classifier.
    Can be used as a baseline or node-wise classifier ignoring graph structure.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5, mlp_hidden: int = 0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        if mlp_hidden and mlp_hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, out_dim),
            )
        else:
            self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None, return_hidden: bool = False):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input features.
            edge_index (torch.Tensor, optional): Ignored, kept for API compatibility with GNNs.
            return_hidden (bool): Return hidden representation.
        """
        h1 = self.fc1(x)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)

        h2 = self.fc2(h1)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = self.dropout(h2)

        logits = self.head(h2)
        if return_hidden:
            return logits, h2
        return logits
