import torch

def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Adds self-loops to the graph.
    
    Args:
        edge_index (torch.Tensor): Edge indices [2, E].
        num_nodes (int): Number of nodes.
        
    Returns:
        torch.Tensor: Edge indices with self-loops [2, E + N].
    """
    loops = torch.arange(num_nodes, device=edge_index.device)
    loops_edge = torch.stack([loops, loops], dim=0)
    return torch.cat([edge_index, loops_edge], dim=1)


def gcn_norm(edge_index: torch.Tensor, num_nodes: int, eps: float = 1e-12, symmetrize_edges: bool = True):
    """
    Computes GCN normalization coefficients.
    
    Args:
        edge_index (torch.Tensor): Edge indices.
        num_nodes (int): Number of nodes.
        eps (float): Numerical stability epsilon.
        symmetrize_edges (bool): Whether to symmetrize edges first.
        
    Returns:
        tuple: (edge_index, edge_weight) where edge_weight is D^(-1/2) A D^(-1/2).
    """
    if symmetrize_edges:
        rev = edge_index[[1, 0]]
        ei = torch.cat([edge_index, rev], dim=1)
    else:
        ei = edge_index
        
    ei = add_self_loops(ei, num_nodes)
    src, dst = ei[0], ei[1]
    
    # Compute degrees (in-degree)
    deg = torch.bincount(dst, minlength=num_nodes).float()
    deg_inv_sqrt = (deg + eps).pow(-0.5)
    
    # Compute weights: deg_inv_sqrt[src] * deg_inv_sqrt[dst]
    weight = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
    
    return ei, weight
