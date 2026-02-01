import torch
from typing import Dict
from src.evaluation.metrics import accuracy, macro_f1

def split_head_tail_by_degree(edge_index: torch.Tensor, num_nodes: int, head_ratio: float = 0.2) -> Dict[str, torch.Tensor]:
    """
    Splits nodes into Head and Tail groups based on degree centrality.
    
    Args:
        edge_index (torch.Tensor): Edge indices.
        num_nodes (int): Total number of nodes.
        head_ratio (float): Ratio of nodes to consider as 'Head'.
        
    Returns:
        Dict[str, torch.Tensor]: Dictionary containing 'head_mask' and 'tail_mask'.
    """
    nodes = torch.cat([edge_index[0], edge_index[1]], dim=0)
    deg = torch.bincount(nodes, minlength=num_nodes)
    order = torch.argsort(deg, descending=True)
    k = max(1, int(num_nodes * head_ratio))
    head_idx = order[:k]
    
    head_mask = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
    head_mask[head_idx] = True
    tail_mask = ~head_mask
    
    return {
        "head_mask": head_mask,
        "tail_mask": tail_mask,
    }

def split_head_tail(labels: torch.Tensor, head_ratio: float = 0.2) -> Dict[str, torch.Tensor]:
    """
    Splits classes into Head and Tail based on label frequency (imbalance).
    """
    labels = labels.view(-1)
    valid_mask = (labels >= 0)
    labels_valid = labels[valid_mask].to(torch.long)
    
    if labels_valid.numel() == 0:
        head_mask = torch.zeros_like(labels, dtype=torch.bool)
        tail_mask = torch.zeros_like(labels, dtype=torch.bool)
        return {
            "head_classes": torch.empty(0, dtype=torch.long, device=labels.device),
            "tail_classes": torch.empty(0, dtype=torch.long, device=labels.device),
            "head_mask": head_mask,
            "tail_mask": tail_mask,
        }
        
    num_classes = int(labels_valid.max().item() + 1)
    counts = torch.bincount(labels_valid, minlength=num_classes)
    order = torch.argsort(counts, descending=True)
    k = max(1, int(counts.numel() * head_ratio))
    head_classes = order[:k]
    
    all_classes = torch.arange(num_classes, device=labels.device)
    mark = torch.zeros_like(all_classes, dtype=torch.bool)
    mark[head_classes] = True
    tail_classes = all_classes[~mark]
    
    head_mask = torch.zeros_like(labels, dtype=torch.bool)
    tail_mask = torch.zeros_like(labels, dtype=torch.bool)
    
    tmp_head = (labels_valid.view(-1, 1) == head_classes.view(1, -1)).any(dim=1)
    tmp_tail = (labels_valid.view(-1, 1) == tail_classes.view(1, -1)).any(dim=1)
    
    head_mask[valid_mask] = tmp_head
    tail_mask[valid_mask] = tmp_tail
    
    return {
        "head_classes": head_classes,
        "tail_classes": tail_classes,
        "head_mask": head_mask,
        "tail_mask": tail_mask,
    }

def compute_group_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    eval_mask: torch.Tensor,
    split_masks: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Computes accuracy and F1 separately for Head and Tail groups.
    """
    device = logits.device
    labels = labels.to(device)
    eval_mask = eval_mask.to(device)
    head_mask = split_masks["head_mask"].to(device)
    tail_mask = split_masks["tail_mask"].to(device)
    
    valid_eval = (labels >= 0)
    head_eval = eval_mask & head_mask & valid_eval
    tail_eval = eval_mask & tail_mask & valid_eval
    
    h_acc = accuracy(logits, labels, head_eval) if head_eval.any() else 0.0
    t_acc = accuracy(logits, labels, tail_eval) if tail_eval.any() else 0.0
    h_f1 = macro_f1(logits, labels, head_eval) if head_eval.any() else 0.0
    t_f1 = macro_f1(logits, labels, tail_eval) if tail_eval.any() else 0.0
    
    return {
        "HeadAcc": h_acc,
        "TailAcc": t_acc,
        "HeadF1": h_f1,
        "TailF1": t_f1,
        "BalancedAcc": (h_acc + t_acc) / 2.0,
    }
