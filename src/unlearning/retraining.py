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
class RetrainConfig:
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

def retrain_after_delete(
    g: GraphData,
    cfg: RetrainConfig,
    delete_node: Optional[int] = None,
    delete_edge: Optional[Tuple[int,int]] = None
):
    t0 = time.time()
    g_mod = g
    if delete_node is not None:
        g_mod = remove_node(g_mod, delete_node, drop_edges=True)
    if delete_edge is not None:
        u, v = delete_edge
        g_mod = remove_edge(g_mod, u, v)

    valid_labels = g_mod.labels[g_mod.labels >= 0]
    num_classes = int(valid_labels.max().item() + 1)
    in_dim = g_mod.num_features

    model = make_model(cfg.model, in_dim, cfg.hidden, num_classes, cfg.dropout, cfg.mlp_hidden, cfg.heads, cfg.symmetrize_edges)
    trainer = Trainer(model, lr=cfg.lr, weight_decay=cfg.wd, device=cfg.device)
    trainer.train(g_mod, epochs=cfg.epochs, early_stop=20)
    train_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        logits = model(g_mod.features.to(cfg.device), g_mod.edge_index.to(cfg.device))
        probs = F.softmax(logits, dim=1)

    y = g_mod.labels.to(cfg.device)
    val_mask = g_mod.val_mask.to(cfg.device)
    test_mask = g_mod.test_mask.to(cfg.device)
    val_acc = accuracy(probs, y, val_mask)
    test_acc = accuracy(probs, y, test_mask)
    val_f1 = macro_f1(probs, y, val_mask)
    test_f1 = macro_f1(probs, y, test_mask)
    
    from src.evaluation.metrics import micro_f1
    val_micro = micro_f1(probs, y, val_mask)
    test_micro = micro_f1(probs, y, test_mask)

    split = split_head_tail_by_degree(g_mod.edge_index, g_mod.num_nodes, head_ratio=cfg.head_ratio)
    val_group = compute_group_metrics(probs, y, val_mask, split)
    test_group = compute_group_metrics(probs, y, test_mask, split)

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
