import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
