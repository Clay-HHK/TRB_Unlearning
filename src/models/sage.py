import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.graph_ops import add_self_loops

class GraphSAGE(nn.Module):
    """
    GraphSAGE implementation with mean aggregation.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5, mlp_hidden: int = 0, symmetrize_edges: bool = True):
        super().__init__()
        # Layer 1 weights: Self + Neighbor
        self.lin_self1 = nn.Linear(in_dim, hidden_dim)
        self.lin_neigh1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Layer 2 weights
        self.lin_self2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin_neigh2 = nn.Linear(hidden_dim, hidden_dim)
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
        self.symmetrize_edges = symmetrize_edges

    def _mean_aggregate(self, ei: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregates neighbor features by averaging.
        """
        N = x.size(0)
        ei = add_self_loops(ei, N)
        src, dst = ei[0], ei[1]
        
        # Calculate in-degree for normalization
        deg_in = torch.bincount(dst, minlength=N).float()
        
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        
        return agg / deg_in.clamp(min=1.0).view(-1, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, return_hidden: bool = False):
        if self.symmetrize_edges:
            rev = edge_index[[1, 0]]
            ei = torch.cat([edge_index, rev], dim=1)
        else:
            ei = edge_index

        # Layer 1
        neigh1 = self._mean_aggregate(ei, x)
        h1 = self.lin_self1(x) + self.lin_neigh1(neigh1)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)

        # Layer 2
        neigh2 = self._mean_aggregate(ei, h1)
        h2 = self.lin_self2(h1) + self.lin_neigh2(neigh2)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = self.dropout(h2)
        
        logits = self.head(h2)
        
        if return_hidden:
            return logits, h2
        return logits
