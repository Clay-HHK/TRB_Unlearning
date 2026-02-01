import numpy as np
import torch
from pathlib import Path
from typing import Optional

def compute_edge_order(g, device: Optional[str] = None) -> torch.Tensor:
    """
    Computes edge order based on the sum of degrees of the connected nodes.
    Used for specific unlearning or processing strategies.
    
    Args:
        g (GraphData): The graph data object containing edge_index and num_nodes.
        device (str, optional): Device to perform computation on.
        
    Returns:
        torch.Tensor: Sorted indices of edges.
    """
    ei = g.edge_index if device is None else g.edge_index.to(device)
    num_nodes = g.num_nodes
    nodes = torch.cat([ei[0], ei[1]], dim=0)
    deg = torch.bincount(nodes, minlength=num_nodes)
    
    # Score = sum of degrees of endpoints
    scores = deg[ei[0]] + deg[ei[1]]
    order = torch.argsort(scores, descending=True)
    return order

def save_edge_order(cache_dir: str | Path, dataset: str, order: torch.Tensor) -> Path:
    """
    Caches the computed edge order to disk.
    """
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    out = p / f"edge_order_{dataset}.npz"
    np.savez(out, order=order.cpu().numpy(), E=int(order.numel()))
    return out

def load_edge_order(cache_dir: str | Path, dataset: str, expected_E: int) -> Optional[torch.Tensor]:
    """
    Loads cached edge order if it matches the expected number of edges.
    """
    p = Path(cache_dir) / f"edge_order_{dataset}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    
    # Handle scalar 0-d array issue if it occurs
    E_cached = int(d["E"].item()) if np.asarray(d["E"]).shape == () else int(np.asarray(d["E"]))
    
    if E_cached != expected_E:
        return None
        
    order_np = np.asarray(d["order"])
    return torch.as_tensor(order_np, dtype=torch.long)
