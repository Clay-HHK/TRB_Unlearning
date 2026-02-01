from dataclasses import dataclass
import torch
import torch.nn.functional as F
import time
from typing import Optional, Tuple
from src.data.graph_data import GraphData
from src.fusion.hctsa import make_model
from src.training.trainer import Trainer
from src.evaluation.metrics import accuracy, macro_f1
from src.evaluation.long_tail import split_head_tail_by_degree, compute_group_metrics
from .utils import remove_node, remove_edge

@dataclass
class DeleteSimilarityConfig:
    model: str = "gcn"
    hidden: int = 128
    mlp_hidden: int = 0
    dropout: float = 0.5
    heads: int = 2
    symmetrize_edges: bool = True
    lr: float = 0.01
    wd: float = 5e-4
    epochs: int = 300
    device: str = "cuda"
    head_ratio: float = 0.2
    alpha: float = 1.0

def _cosine_similarity_loss(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(a[mask], b[mask]).mean()

def delete_similarity_train(
    g: GraphData,
    cfg: DeleteSimilarityConfig,
    delete_node: Optional[int] = None,
    delete_edge: Optional[Tuple[int,int]] = None,
):
    g_forget = g
    if delete_node is not None:
        g_forget = remove_node(g_forget, delete_node, drop_edges=True)
    if delete_edge is not None:
        u, v = delete_edge
        g_forget = remove_edge(g_forget, u, v)

    valid_labels = g.labels[g.labels >= 0]
    num_classes = int(valid_labels.max().item() + 1)
    in_dim = g.num_features

    model = make_model(cfg.model, in_dim, cfg.hidden, num_classes, cfg.dropout, cfg.mlp_hidden, cfg.heads, cfg.symmetrize_edges)
    device = cfg.device
    trainer = Trainer(model, lr=cfg.lr, weight_decay=cfg.wd, device=device)

    x = g.features.to(device)
    ei = g.edge_index.to(device)
    y = g.labels.to(device)
    train_mask = g.train_mask.to(device)
    val_mask = g.val_mask.to(device)
    test_mask = g.test_mask.to(device)

    x_f = g_forget.features.to(device)
    ei_f = g_forget.edge_index.to(device)

    best_state = None
    best_val = -1.0
    val_no_improve = 0
    t0 = time.time()
    
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        trainer.opt.zero_grad()

        with torch.no_grad():
            logits_orig = model(x, ei)

        logits_forget = model(x_f, ei_f)
        loss_mask = train_mask & (y >= 0)
        
        loss_ce = F.cross_entropy(logits_forget[loss_mask], y[loss_mask]) if loss_mask.any() else torch.tensor(0.0, device=device)
        loss_sim = _cosine_similarity_loss(logits_orig.detach(), logits_forget, train_mask)
        loss = loss_ce + cfg.alpha * loss_sim
        
        loss.backward()
        trainer.opt.step()

        model.eval()
        with torch.no_grad():
            logits_val = model(x, ei)
            val_acc = accuracy(logits_val, y, val_mask)
            
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
            val_no_improve = 0
        else:
            val_no_improve += 1
            
        if val_no_improve >= 20:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        logits = model(x_f, ei_f)
        probs = F.softmax(logits, dim=1)

    y_f = g_forget.labels.to(device)
    val_mask_f = g_forget.val_mask.to(device)
    test_mask_f = g_forget.test_mask.to(device)
    
    val_acc = accuracy(probs, y_f, val_mask_f)
    test_acc = accuracy(probs, y_f, test_mask_f)
    val_f1 = macro_f1(probs, y_f, val_mask_f)
    test_f1 = macro_f1(probs, y_f, test_mask_f)
    
    from src.evaluation.metrics import micro_f1
    val_micro = micro_f1(probs, y_f, val_mask_f)
    test_micro = micro_f1(probs, y_f, test_mask_f)
    
    split = split_head_tail_by_degree(g_forget.edge_index, g_forget.num_nodes, head_ratio=cfg.head_ratio)
    val_group = compute_group_metrics(probs, y_f, val_mask_f, split)
    test_group = compute_group_metrics(probs, y_f, test_mask_f, split)

    train_time = float(time.time() - t0)
    
    return {
      "model": model,
      "probs": probs,
      "ValAcc": float(val_acc),
      "TestAcc": float(test_acc),
      "ValMicroF1": float(val_micro),
      "TestMicroF1": float(test_micro),
      "ValF1": float(val_f1),
      "TestF1": float(test_f1),
      "ValHeadAcc": float(val_group["HeadAcc"]),
      "ValTailAcc": float(val_group["TailAcc"]),
      "TestHeadAcc": float(test_group["HeadAcc"]),
      "TestTailAcc": float(test_group["TailAcc"]),
      "TrainTime": float(train_time),
    }
