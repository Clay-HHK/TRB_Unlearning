import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.graph_ops import gcn_norm

def scatter_add_messages(edge_index: torch.Tensor, edge_weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Aggregates messages from neighbors using scatter_add (simulated via index_add_).
    
    Args:
        edge_index (torch.Tensor): Edge indices [2, E].
        edge_weight (torch.Tensor): Edge weights [E].
        x (torch.Tensor): Node features [N, D].
        
    Returns:
        torch.Tensor: Aggregated node features [N, D].
    """
    src, dst = edge_index[0], edge_index[1]
    # Message calculation: source node features * edge weights
    msg = x[src] * edge_weight.view(-1, 1)
    out = torch.zeros_like(x)
    # Aggregate messages to destination nodes
    out.index_add_(0, dst, msg)
    return out

class SimpleGCN(nn.Module):
    """
    A simple Graph Convolutional Network (GCN) implementation.
    
    Attributes:
        lin1 (nn.Linear): First linear layer.
        bn1 (nn.BatchNorm1d): Batch normalization for the first layer.
        lin2 (nn.Linear): Second linear layer.
        bn2 (nn.BatchNorm1d): Batch normalization for the second layer.
        dropout (nn.Dropout): Dropout layer.
        head (nn.Module): Output classification head (Linear or MLP).
        symmetrize_edges (bool): Whether to symmetrize the input adjacency matrix.
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5, mlp_hidden: int = 0, symmetrize_edges: bool = True):
        """
        Initializes the SimpleGCN model.

        Args:
            in_dim (int): Input feature dimension.
            hidden_dim (int): Hidden feature dimension.
            out_dim (int): Output dimension (number of classes).
            dropout (float): Dropout probability. Default is 0.5.
            mlp_hidden (int): Hidden dimension for the MLP head. If 0, uses a single Linear layer. Default is 0.
            symmetrize_edges (bool): Whether to automatically symmetrize edges during forward pass. Default is True.
        """
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.symmetrize_edges = symmetrize_edges

        if mlp_hidden and mlp_hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, out_dim),
            )
        else:
            self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, return_hidden: bool = False):
        """
        Forward pass of the GCN.

        Args:
            x (torch.Tensor): Node feature matrix [N, F].
            edge_index (torch.Tensor): Graph connectivity [2, E].
            return_hidden (bool): If True, returns the hidden representation before the head.

        Returns:
            torch.Tensor or tuple: Logits [N, C], or (logits, hidden_rep) if return_hidden is True.
        """
        num_nodes = x.size(0)
        # Compute GCN normalization (symmetrization handled here if enabled)
        ei, w = gcn_norm(edge_index, num_nodes, symmetrize_edges=self.symmetrize_edges)

        # Layer 1
        x = self.lin1(x)
        x = scatter_add_messages(ei, w, x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.lin2(x)
        x = scatter_add_messages(ei, w, x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        h = x
        logits = self.head(h)
        
        if return_hidden:
            return logits, h
        return logits
