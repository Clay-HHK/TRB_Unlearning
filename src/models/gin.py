import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.graph_ops import add_self_loops

class _MLP(nn.Module):
    """
    Helper MLP class for GIN layers.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class GIN(nn.Module):
    """
    Graph Isomorphism Network (GIN).
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5, mlp_hidden: int = 0, symmetrize_edges: bool = True):
        super().__init__()
        self.eps1 = nn.Parameter(torch.zeros(1))
        self.eps2 = nn.Parameter(torch.zeros(1))

        # Input projection to hidden dimension
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.bn_in = nn.BatchNorm1d(hidden_dim)

        # Layer 1 MLP
        self.mlp1 = _MLP(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Layer 2 MLP
        self.mlp2 = _MLP(hidden_dim, hidden_dim, hidden_dim, dropout)
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

    def _sum_aggregate(self, ei: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Sum aggregation for GIN.
        """
        N = x.size(0)
        ei = add_self_loops(ei, N)
        src, dst = ei[0], ei[1]
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        return agg

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, return_hidden: bool = False):
        if self.symmetrize_edges:
            rev = edge_index[[1, 0]]
            ei = torch.cat([edge_index, rev], dim=1)
        else:
            ei = edge_index

        # Pre-projection
        x0 = self.input_proj(x)
        x0 = self.bn_in(x0)
        x0 = F.relu(x0)

        # Layer 1
        agg1 = self._sum_aggregate(ei, x0)
        h1 = self.mlp1((1 + self.eps1) * x0 + agg1)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)

        # Layer 2
        agg2 = self._sum_aggregate(ei, h1)
        h2 = self.mlp2((1 + self.eps2) * h1 + agg2)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = self.dropout(h2)
        
        h = h2
        logits = self.head(h)
        
        if return_hidden:
            return logits, h
        return logits
