import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.graph_ops import add_self_loops

def _segment_softmax(dst: torch.Tensor, logits: torch.Tensor, num_nodes: int, eps: float = 1e-12) -> torch.Tensor:
    """
    Computes segment softmax (softmax over neighbors for each node).
    
    Args:
        dst (torch.Tensor): Destination node indices [E].
        logits (torch.Tensor): Unnormalized attention scores [E].
        num_nodes (int): Total number of nodes.
        eps (float): Epsilon for numerical stability.
        
    Returns:
        torch.Tensor: Softmax scores [E].
    """
    exp_logits = torch.exp(logits)
    denom = torch.zeros(num_nodes, device=logits.device)
    # Sum exponentials for each destination node
    denom.index_add_(0, dst, exp_logits)
    return exp_logits / (denom[dst] + eps)

class SimpleGAT(nn.Module):
    """
    A simple Graph Attention Network (GAT) implementation.
    
    Attributes:
        heads (int): Number of attention heads.
        out_per_head (int): Output dimension per head.
        hidden_dim (int): Total hidden dimension (heads * out_per_head).
        lin1 (nn.Linear): Input projection for layer 1.
        a_l1, a_r1 (nn.Parameter): Attention vectors for layer 1.
        lin2 (nn.Linear): Input projection for layer 2.
        a_l2, a_r2 (nn.Parameter): Attention vectors for layer 2.
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.5, mlp_hidden: int = 0, symmetrize_edges: bool = True):
        """
        Initializes the SimpleGAT model.

        Args:
            in_dim (int): Input feature dimension.
            hidden_dim (int): Hidden feature dimension (will be adjusted to be divisible by heads).
            out_dim (int): Output dimension.
            heads (int): Number of attention heads. Default is 4.
            dropout (float): Dropout probability. Default is 0.5.
            mlp_hidden (int): Hidden dimension for the MLP head.
            symmetrize_edges (bool): Whether to symmetrize edges.
        """
        super().__init__()
        assert hidden_dim > 0 and heads > 0
        self.heads = heads
        self.out_per_head = max(1, hidden_dim // heads)
        self.hidden_dim = self.out_per_head * heads
        self.symmetrize_edges = symmetrize_edges

        # Layer 1
        self.lin1 = nn.Linear(in_dim, self.hidden_dim)
        self.a_l1 = nn.Parameter(torch.empty(heads, self.out_per_head))
        self.a_r1 = nn.Parameter(torch.empty(heads, self.out_per_head))
        nn.init.xavier_uniform_(self.a_l1)
        nn.init.xavier_uniform_(self.a_r1)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)

        # Layer 2
        self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.a_l2 = nn.Parameter(torch.empty(heads, self.out_per_head))
        self.a_r2 = nn.Parameter(torch.empty(heads, self.out_per_head))
        nn.init.xavier_uniform_(self.a_l2)
        nn.init.xavier_uniform_(self.a_r2)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim)

        self.dropout = nn.Dropout(dropout)

        if mlp_hidden and mlp_hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(self.hidden_dim, mlp_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, out_dim),
            )
        else:
            self.head = nn.Linear(self.hidden_dim, out_dim)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def _attn_aggregate(self, ei: torch.Tensor, x_proj: torch.Tensor, a_l: torch.Tensor, a_r: torch.Tensor) -> torch.Tensor:
        """
        Computes multi-head attention aggregation.
        """
        # Reshape projection to [N, heads, d]
        N = x_proj.size(0)
        H = x_proj.view(N, self.heads, self.out_per_head)
        src, dst = ei[0], ei[1]
        out = torch.zeros(N, self.heads, self.out_per_head, device=x_proj.device)

        # Compute attention per head
        # We iterate over heads to avoid creating a massive [E, heads, d] tensor
        for i in range(self.heads):
            h_src_i = H[src, i, :]              # [E, d]
            h_dst_i = H[dst, i, :]              # [E, d]
            # Attention score: a^T [Wh_i || Wh_j]
            logits_i = (h_src_i * a_l[i].view(1, -1)).sum(dim=-1) + \
                       (h_dst_i * a_r[i].view(1, -1)).sum(dim=-1)   # [E]
            alpha_i = _segment_softmax(dst, logits_i, N)            # [E]
            
            msg_i = h_src_i * alpha_i.view(-1, 1)                   # [E, d]
            out[:, i, :].index_add_(0, dst, msg_i)                  # [N, d] += [E, d]

        return out.view(N, self.heads * self.out_per_head)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, return_hidden: bool = False):
        """
        Forward pass of GAT.
        """
        # Symmetrize + self-loops
        if self.symmetrize_edges:
            rev = edge_index[[1, 0]]
            ei = torch.cat([edge_index, rev], dim=1)
        else:
            ei = edge_index
        ei = add_self_loops(ei, x.size(0))

        # Layer 1
        x1 = self.lin1(x)
        x1 = self._attn_aggregate(ei, x1, self.a_l1, self.a_r1)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)

        # Layer 2
        x2 = self.lin2(x1)
        x2 = self._attn_aggregate(ei, x2, self.a_l2, self.a_r2)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = self.dropout(x2)
        
        h = x2
        logits = self.head(h)
        
        if return_hidden:
            return logits, h
        return logits
