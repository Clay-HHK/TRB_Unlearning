import torch
import torch.nn.functional as F
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from src.data.graph_data import GraphData
from src.fusion.hctsa import make_model, build_subgraph
from src.training.trainer import Trainer
from src.evaluation.metrics import accuracy, macro_f1
from src.evaluation.long_tail import split_head_tail_by_degree, compute_group_metrics

@dataclass
class BEKMConfig:
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
    K: int = 4
    seed: int = 42
    head_ratio: float = 0.2

def _balanced_kmeans_assign(features: torch.Tensor, K: int, seed: int = 42) -> torch.Tensor:
    orig_dev = features.device
    dev = orig_dev
    if torch.cuda.is_available() and dev.type != "cuda":
        dev = torch.device("cuda")
    try:
        from sklearn.cluster import KMeans
        x = features.detach().cpu().numpy()
        km = KMeans(n_clusters=K, random_state=seed, n_init=10)
        km.fit(x)
        centers = torch.from_numpy(km.cluster_centers_).to(dev, dtype=features.dtype)
        x_dev = features.to(dev)
        D = torch.cdist(x_dev, centers)  # [N, K] on GPU when available
        N = features.size(0)
        quota = N // K
        extra = N - quota * K
        assigned = torch.full((N,), -1, dtype=torch.long, device=dev)
        used = torch.zeros(K, dtype=torch.long, device=dev)
        orders = [torch.argsort(D[:, i], descending=False) for i in range(K)]
        ptrs = [0 for _ in range(K)]
        for i in range(K):
            need = quota + (1 if i < extra else 0)
            count = 0
            while count < need and ptrs[i] < N:
                u = int(orders[i][ptrs[i]].item()); ptrs[i] += 1
                if int(assigned[u].item()) == -1:
                    assigned[u] = i
                    used[i] += 1
                    count += 1
        nearest = torch.argmin(D, dim=1)
        for u in range(N):
            if int(assigned[u].item()) != -1:
                continue
            cand = int(nearest[u].item())
            best = cand
            min_used = int(used[best].item())
            for j in range(K):
                if int(used[j].item()) < min_used:
                    best = j; min_used = int(used[j].item())
            assigned[u] = best
            used[best] += 1
        return assigned.to(orig_dev)
    except Exception:
        N = features.size(0)
        perm = torch.randperm(N, device=orig_dev)
        assigned = torch.full((N,), -1, dtype=torch.long, device=orig_dev)
        quota = N // K
        extra = N - quota * K
        ptr = 0
        for i in range(K):
            need = quota + (1 if i < extra else 0)
            idx = perm[ptr:ptr+need]; ptr += need
            assigned[idx] = i
        return assigned

class BEKMUnlearningManager:
    def __init__(self, g: GraphData, cfg: BEKMConfig):
        self.g = g
        self.cfg = cfg
        self.assign = _balanced_kmeans_assign(g.features, cfg.K, seed=cfg.seed)
        self.subgraphs: List[GraphData] = []
        self.node_indices: List[torch.Tensor] = []
        for i in range(cfg.K):
            mask = (self.assign == i)
            sg, idx = build_subgraph(g, mask)
            self.subgraphs.append(sg)
            self.node_indices.append(idx)
        valid_labels = g.labels[g.labels >= 0]
        self.num_classes = int(valid_labels.max().item() + 1)
        self.in_dim = g.num_features
        self.models: List[torch.nn.Module] = []
        self.pred_list: List[torch.Tensor] = []

    def _train_subgraph(self, i: int, epochs: Optional[int] = None):
        epochs = epochs or self.cfg.epochs
        t0 = time.time()
        model = make_model(self.cfg.model, self.in_dim, self.cfg.hidden, self.num_classes, self.cfg.dropout, self.cfg.mlp_hidden, self.cfg.heads, self.cfg.symmetrize_edges)
        trainer = Trainer(model, lr=self.cfg.lr, weight_decay=self.cfg.wd, device=self.cfg.device)
        trainer.train(self.subgraphs[i], epochs=epochs, early_stop=20)
        train_time = time.time() - t0
        model.eval()
        with torch.no_grad():
            logits = model(self.subgraphs[i].features.to(self.cfg.device), self.subgraphs[i].edge_index.to(self.cfg.device))
            probs = F.softmax(logits, dim=1)
        self.models[i] = model
        self.pred_list[i] = probs
        return float(train_time)

    def train_initial(self):
        self.models = []
        self.pred_list = []
        for i in range(self.cfg.K):
            model = make_model(self.cfg.model, self.in_dim, self.cfg.hidden, self.num_classes, self.cfg.dropout, self.cfg.mlp_hidden, self.cfg.heads, self.cfg.symmetrize_edges)
            trainer = Trainer(model, lr=self.cfg.lr, weight_decay=self.cfg.wd, device=self.cfg.device)
            t0 = time.time()
            trainer.train(self.subgraphs[i], epochs=self.cfg.epochs, early_stop=20)
            model.eval()
            with torch.no_grad():
                logits = model(self.subgraphs[i].features.to(self.cfg.device), self.subgraphs[i].edge_index.to(self.cfg.device))
                probs = F.softmax(logits, dim=1)
            self.models.append(model)
            self.pred_list.append(probs)
        return

    def fused_probs(self) -> torch.Tensor:
        N = self.g.num_nodes
        C = self.num_classes
        fused = torch.zeros(N, C, device=self.cfg.device)
        counts = torch.zeros(N, dtype=torch.float32, device=self.cfg.device)
        for i in range(self.cfg.K):
            idx = self.node_indices[i]
            p = self.pred_list[i].to(self.cfg.device)
            fused[idx] += p
            counts[idx] += 1.0
        counts = torch.clamp(counts, min=1.0)
        fused = fused / counts.view(-1, 1)
        return fused

    def delete_node(self, u: int):
        shard = int(self.assign[u].item())
        sg = self.subgraphs[shard]
        idx_local = torch.nonzero(self.node_indices[shard] == u, as_tuple=False).view(-1)
        if idx_local.numel() == 0:
            return 0.0
        u_local = int(idx_local.item())
        ei = sg.edge_index
        keep = (ei[0] != u_local) & (ei[1] != u_local)
        new_ei = ei[:, keep]
        new_labels = sg.labels.clone()
        new_train = sg.train_mask.clone()
        new_labels[u_local] = -1
        new_train[u_local] = False
        self.subgraphs[shard] = GraphData(
            features=sg.features,
            labels=new_labels,
            edge_index=new_ei,
            train_mask=new_train,
            val_mask=sg.val_mask,
            test_mask=sg.test_mask,
        )
        retrain_time = self._train_subgraph(shard, epochs=max(50, self.cfg.epochs // 4))
        return float(retrain_time)

    def delete_edge(self, u: int, v: int):
        su = int(self.assign[u].item()); sv = int(self.assign[v].item())
        if su != sv:
            return 0.0
        shard = su
        sg = self.subgraphs[shard]
        idx_u = torch.nonzero(self.node_indices[shard] == u, as_tuple=False).view(-1)
        idx_v = torch.nonzero(self.node_indices[shard] == v, as_tuple=False).view(-1)
        if idx_u.numel() == 0 or idx_v.numel() == 0:
            return 0.0
        uu = int(idx_u.item()); vv = int(idx_v.item())
        ei = sg.edge_index
        mask = ~(((ei[0] == uu) & (ei[1] == vv)) | ((ei[0] == vv) & (ei[1] == uu)))
        new_ei = ei[:, mask]
        self.subgraphs[shard] = GraphData(
            features=sg.features,
            labels=sg.labels,
            edge_index=new_ei,
            train_mask=sg.train_mask,
            val_mask=sg.val_mask,
            test_mask=sg.test_mask,
        )
        retrain_time = self._train_subgraph(shard, epochs=max(50, self.cfg.epochs // 4))
        return float(retrain_time)

    def evaluate(self):
        t0 = time.time()
        probs = self.fused_probs()
        fusion_time = time.time() - t0
        y = self.g.labels.to(self.cfg.device)
        val_mask = self.g.val_mask.to(self.cfg.device)
        test_mask = self.g.test_mask.to(self.cfg.device)
        val_acc = accuracy(probs, y, val_mask)
        test_acc = accuracy(probs, y, test_mask)
        val_f1 = macro_f1(probs, y, val_mask)
        test_f1 = macro_f1(probs, y, test_mask)
        split = split_head_tail_by_degree(self.g.edge_index, self.g.num_nodes, head_ratio=self.cfg.head_ratio)
        val_group = compute_group_metrics(probs, y, val_mask, split)
        test_group = compute_group_metrics(probs, y, test_mask, split)
        return {
          "probs": probs,
          "ValAcc": float(val_acc),
          "TestAcc": float(test_acc),
          "ValF1": float(val_f1),
          "TestF1": float(test_f1),
          "ValHeadAcc": float(val_group["HeadAcc"]),
          "ValTailAcc": float(val_group["TailAcc"]),
          "TestHeadAcc": float(test_group["HeadAcc"]),
          "TestTailAcc": float(test_group["TailAcc"]),
          "FusionTime": float(fusion_time),
        }
