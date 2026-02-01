from dataclasses import dataclass
import torch
import torch.nn.functional as F
import time
from typing import Optional, Tuple, List
from src.data.graph_data import GraphData
from src.fusion.hctsa import make_model
from src.training.trainer import Trainer
from src.evaluation.metrics import accuracy, macro_f1
from src.evaluation.long_tail import split_head_tail_by_degree, compute_group_metrics

@dataclass
class InfluenceConfig:
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
    damping: float = 1e-3
    scale: float = 1.0

def _params(model) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]

def _model_logits(model, g: GraphData, device: str) -> torch.Tensor:
    return model(g.features.to(device), g.edge_index.to(device))

def influence_unlearn(
    g: GraphData,
    cfg: InfluenceConfig,
    delete_node: Optional[int] = None,
    delete_edge: Optional[Tuple[int,int]] = None
):
    valid_labels = g.labels[g.labels >= 0]
    num_classes = int(valid_labels.max().item() + 1)
    in_dim = g.num_features

    model = make_model(cfg.model, in_dim, cfg.hidden, num_classes, cfg.dropout, cfg.mlp_hidden, cfg.heads, cfg.symmetrize_edges)
    trainer = Trainer(model, lr=cfg.lr, weight_decay=cfg.wd, device=cfg.device)
    t_train = time.time()
    trainer.train(g, epochs=cfg.epochs, early_stop=20)
    train_time = time.time() - t_train

    model.eval()
    params = _params(model)

    logits = _model_logits(model, g, cfg.device)
    y = g.labels.to(cfg.device)
    train_mask = g.train_mask.to(cfg.device)
    n_train = int(train_mask.sum().item())

    t_upd = time.time()
    loss_train = F.cross_entropy(logits[train_mask], y[train_mask])
    grads_train = torch.autograd.grad(loss_train, params, retain_graph=True)

    remove_mask = torch.zeros(g.num_nodes, dtype=torch.bool, device=cfg.device)
    if delete_node is not None:
        remove_mask[delete_node] = True
    if delete_edge is not None:
        u, v = delete_edge
        remove_mask[u] = True
        remove_mask[v] = True
    remove_mask = remove_mask & train_mask

    if remove_mask.sum().item() == 0:
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
        val_mask = g.val_mask.to(cfg.device)
        test_mask = g.test_mask.to(cfg.device)
        val_acc = accuracy(probs, y, val_mask)
        test_acc = accuracy(probs, y, test_mask)
        val_f1 = macro_f1(probs, y, val_mask)
        test_f1 = macro_f1(probs, y, test_mask)
        split = split_head_tail_by_degree(g.edge_index, g.num_nodes, head_ratio=cfg.head_ratio)
        val_group = compute_group_metrics(probs, y, val_mask, split)
        test_group = compute_group_metrics(probs, y, test_mask, split)
        return {
          "model": model,
          "probs": probs,
          "ValAcc": float(val_acc),
          "TestAcc": float(test_acc),
          "ValF1": float(val_f1),
          "TestF1": float(test_f1),
          "ValHeadAcc": float(val_group["HeadAcc"]),
          "ValTailAcc": float(val_group["TailAcc"]),
          "TestHeadAcc": float(test_group["HeadAcc"]),
          "TestTailAcc": float(test_group["TailAcc"]),
          "TrainTime": float(train_time),
          "UpdateTime": 0.0,
        }

    loss_rem = F.cross_entropy(logits[remove_mask], y[remove_mask])
    grads_rem = torch.autograd.grad(loss_rem, params, retain_graph=False)
    for p, g_tr, g_rm in zip(params, grads_train, grads_rem):
        fisher_diag = g_tr.detach()**2 + cfg.damping
        delta = - cfg.scale * (g_rm.detach() / fisher_diag)
        delta = delta / max(1, n_train)
        p.data.add_(delta)
    update_time = float(time.time() - t_upd)

    g_eval = g
    if delete_edge is not None:
        ei = g.edge_index
        u, v = delete_edge
        mask = ~(((ei[0] == u) & (ei[1] == v)) | ((ei[0] == v) & (ei[1] == u)))
        new_ei = ei[:, mask]
        g_eval = GraphData(g.features, g.labels, new_ei, g.train_mask, g.val_mask, g.test_mask)

    with torch.no_grad():
        logits2 = _model_logits(model, g_eval, cfg.device)
        probs = F.softmax(logits2, dim=1)

    val_mask = g_eval.val_mask.to(cfg.device)
    test_mask = g_eval.test_mask.to(cfg.device)
    val_acc = accuracy(probs, y, val_mask)
    test_acc = accuracy(probs, y, test_mask)
    val_f1 = macro_f1(probs, y, val_mask)
    test_f1 = macro_f1(probs, y, test_mask)
    
    from src.evaluation.metrics import micro_f1
    val_micro = micro_f1(probs, y, val_mask)
    test_micro = micro_f1(probs, y, test_mask)
    
    split = split_head_tail_by_degree(g_eval.edge_index, g_eval.num_nodes, head_ratio=cfg.head_ratio)
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
      "UpdateTime": float(update_time),
    }
