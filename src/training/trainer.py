from dataclasses import dataclass
import torch
import torch.nn.functional as F

from src.evaluation.metrics import accuracy, macro_f1
from src.evaluation.long_tail import split_head_tail_by_degree, compute_group_metrics

@dataclass
class TrainResult:
    """
    Data class to store training results.
    """
    best_val_acc: float
    best_test_acc: float
    best_val_f1: float
    best_test_f1: float
    epochs_run: int
    
    # Long-tail metrics
    best_val_head_acc: float
    best_val_tail_acc: float
    best_test_head_acc: float
    best_test_tail_acc: float
    best_val_head_f1: float
    best_val_tail_f1: float
    best_test_head_f1: float
    best_test_tail_f1: float
    best_val_balanced_acc: float
    best_test_balanced_acc: float


class Trainer:
    """
    Generic Trainer class for GNN models.
    """
    def __init__(self, model: torch.nn.Module, lr: float = 0.01, weight_decay: float = 0.0, device: str = "cpu"):
        self.model = model.to(device)
        self.opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = torch.device(device)

    def train(self, g, epochs: int = 200, early_stop: int = 20) -> TrainResult:
        """
        Runs the training loop with early stopping.
        
        Args:
            g (GraphData): Graph data object.
            epochs (int): Maximum epochs.
            early_stop (int): Early stopping patience.
            
        Returns:
            TrainResult: The results of the best validation epoch.
        """
        x = g.features.to(self.device)
        y = g.labels.to(self.device)
        ei = g.edge_index.to(self.device)
        train_mask = g.train_mask.to(self.device)
        val_mask = g.val_mask.to(self.device)
        test_mask = g.test_mask.to(self.device)

        best_state = None
        best_val = -1.0
        val_no_improve = 0
        
        # Track metrics for the best validation step
        best_val_acc = 0.0
        best_test_acc = 0.0
        best_val_f1 = 0.0
        best_test_f1 = 0.0
        val_group = {}
        test_group = {}
        epochs_run = 0

        for epoch in range(1, epochs + 1):
            epochs_run = epoch
            self.model.train()
            self.opt.zero_grad()
            logits = self.model(x, ei)
            loss_mask = train_mask & (y >= 0)
            
            if loss_mask.any():
                loss = F.cross_entropy(logits[loss_mask], y[loss_mask])
                loss.backward()
                self.opt.step()
            else:
                loss = torch.tensor(0.0, device=self.device)

            self.model.eval()
            with torch.no_grad():
                logits = self.model(x, ei)
                val_acc = accuracy(logits, y, val_mask)
                
                # Check for improvement
                if val_acc > best_val:
                    best_val = val_acc
                    best_state = {k: v.clone().detach().cpu() for k, v in self.model.state_dict().items()}
                    val_no_improve = 0
                else:
                    val_no_improve += 1

            if val_no_improve >= early_stop:
                break

        # Load best model
        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        # Final evaluation on best model
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x, ei)
            best_val_acc = accuracy(logits, y, val_mask)
            best_test_acc = accuracy(logits, y, test_mask)
            best_val_f1 = macro_f1(logits, y, val_mask)
            best_test_f1 = macro_f1(logits, y, test_mask)
            
            # Long-tail analysis
            split = split_head_tail_by_degree(ei, g.num_nodes, head_ratio=0.2)
            val_group = compute_group_metrics(logits, y, val_mask, split)
            test_group = compute_group_metrics(logits, y, test_mask, split)

        return TrainResult(
            best_val_acc=best_val_acc,
            best_test_acc=best_test_acc,
            best_val_f1=best_val_f1,
            best_test_f1=best_test_f1,
            epochs_run=epochs_run,
            best_val_head_acc=val_group.get("HeadAcc", 0.0),
            best_val_tail_acc=val_group.get("TailAcc", 0.0),
            best_test_head_acc=test_group.get("HeadAcc", 0.0),
            best_test_tail_acc=test_group.get("TailAcc", 0.0),
            best_val_head_f1=val_group.get("HeadF1", 0.0),
            best_val_tail_f1=val_group.get("TailF1", 0.0),
            best_test_head_f1=test_group.get("HeadF1", 0.0),
            best_test_tail_f1=test_group.get("TailF1", 0.0),
            best_val_balanced_acc=val_group.get("BalancedAcc", 0.0),
            best_test_balanced_acc=test_group.get("BalancedAcc", 0.0),
        )
