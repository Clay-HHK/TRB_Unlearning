import torch

def accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Computes accuracy for the masked nodes.
    
    Args:
        logits (torch.Tensor): Logits [N, C].
        labels (torch.Tensor): Ground truth labels [N].
        mask (torch.Tensor): Boolean mask [N].
        
    Returns:
        float: Accuracy.
    """
    if mask.sum().item() == 0:
        return 0.0
    pred = logits.argmax(dim=1)
    correct = (pred[mask] == labels[mask]).sum().item()
    total = int(mask.sum().item())
    return correct / max(total, 1)


def macro_f1(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Computes Macro-F1 score for the masked nodes.
    
    Args:
        logits (torch.Tensor): Logits [N, C].
        labels (torch.Tensor): Ground truth labels [N].
        mask (torch.Tensor): Boolean mask [N].
        
    Returns:
        float: Macro-F1 score.
    """
    if mask.sum().item() == 0:
        return 0.0
    pred = logits.argmax(dim=1)
    y_true = labels[mask]
    y_pred = pred[mask]
    classes = torch.unique(y_true).tolist()
    
    f1s = []
    for c in classes:
        c = int(c)
        tp = ((y_pred == c) & (y_true == c)).sum().item()
        fp = ((y_pred == c) & (y_true != c)).sum().item()
        fn = ((y_pred != c) & (y_true == c)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1s.append(f1)
        
    return sum(f1s) / max(len(f1s), 1)


def micro_f1(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Computes Micro-F1 score (equivalent to accuracy in single-label multiclass).
    
    Args:
        logits (torch.Tensor): Logits [N, C].
        labels (torch.Tensor): Ground truth labels [N].
        mask (torch.Tensor): Boolean mask [N].
        
    Returns:
        float: Micro-F1 score.
    """
    if mask.sum().item() == 0:
        return 0.0
    valid = (labels >= 0) & mask
    total = int(valid.sum().item())
    if total == 0:
        return 0.0
    pred = logits.argmax(dim=1)
    correct = (pred[valid] == labels[valid]).sum().item()
    return correct / max(total, 1)
