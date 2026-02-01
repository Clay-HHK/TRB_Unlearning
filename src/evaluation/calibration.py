import torch
import torch.nn.functional as F

def temperature_scale_from_val(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, max_iter: int = 50) -> float:
    """
    Learns a temperature scaling factor T using validation data to calibrate logits.
    
    Args:
        logits (torch.Tensor): Unnormalized logits [N, C].
        labels (torch.Tensor): Ground truth labels [N].
        mask (torch.Tensor): Validation mask [N].
        max_iter (int): Maximum LBFGS iterations.
        
    Returns:
        float: Learned temperature T.
    """
    idx = torch.nonzero(mask, as_tuple=False).view(-1)
    if idx.numel() == 0:
        return 1.0
    x = logits[idx]
    y = labels[idx]
    T = torch.ones(1, device=logits.device, requires_grad=True)
    optim = torch.optim.LBFGS([T], lr=0.01, max_iter=max_iter)

    def closure():
        optim.zero_grad()
        loss = F.cross_entropy(x / T, y)
        loss.backward()
        return loss

    optim.step(closure)
    T_val = float(T.detach().clamp(0.5, 5.0).item())
    return T_val

def apply_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:
    """
    Applies temperature scaling to logits.
    
    Args:
        logits (torch.Tensor): Logits [N, C].
        T (float): Temperature.
        
    Returns:
        torch.Tensor: Calibrated probabilities (softmax output).
    """
    return F.softmax(logits / max(T, 1e-6), dim=1)
