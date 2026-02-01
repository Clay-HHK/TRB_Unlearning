import torch
from typing import List, Tuple, Optional
import torch.nn.functional as F
import time

from src.data.graph_data import GraphData
from src.fusion.hctsa import SubgraphAssignments, compute_struct_weights, fuse_predictions
from src.training.trainer import Trainer

class HCTSAUnlearningManager:
    def __init__(
        self,
        g: GraphData,
        subgraphs: List[GraphData],
        assignments: SubgraphAssignments,
        models: List[torch.nn.Module],
        device: str,
        num_classes: int,
    ):
        self.g = g
        self.subgraphs = subgraphs
        self.assign = assignments
        self.models = models
        self.device = device
        self.num_classes = num_classes
        self.last_affected_indices = []

    def _remove_node_in_subgraph(self, sg: GraphData, u: int) -> GraphData:
        ei = sg.edge_index
        keep = (ei[0] != u) & (ei[1] != u)
        new_ei = ei[:, keep]
        new_labels = sg.labels.clone()
        new_train = sg.train_mask.clone()
        new_labels[u] = -1            # Mark as invalid
        new_train[u] = False          # Exclude from training
        return GraphData(
            features=sg.features,
            labels=new_labels,
            edge_index=new_ei,
            train_mask=new_train,
            val_mask=sg.val_mask,
            test_mask=sg.test_mask,
        )

    def _remove_edge_in_subgraph(self, sg: GraphData, u: int, v: int) -> GraphData:
        ei = sg.edge_index
        mask = ~(((ei[0] == u) & (ei[1] == v)) | ((ei[0] == v) & (ei[1] == u)))
        new_ei = ei[:, mask]
        return GraphData(
            features=sg.features,
            labels=sg.labels,
            edge_index=new_ei,
            train_mask=sg.train_mask,
            val_mask=sg.val_mask,
            test_mask=sg.test_mask,
        )

    def delete_node(self, u: int, lr: float, wd: float, epochs: int, early_stop: int = 20) -> Tuple[List[torch.Tensor], List[torch.Tensor], float, float]:
        # Identify affected subgraphs
        aff = self.assign.node_aff_subgraphs[u]
        affected_indices = torch.nonzero(aff, as_tuple=False).view(-1).tolist()
        self.last_affected_indices = affected_indices
        
        if len(affected_indices) == 0:
            return [], [], 0.0, 0.0

        # Update subgraphs
        t_update = time.time()
        for i in affected_indices:
            self.subgraphs[i] = self._remove_node_in_subgraph(self.subgraphs[i], u)
        update_time = float(time.time() - t_update)

        # Retrain affected models
        new_logits, new_hidden = [], []
        retrain_time = 0.0
        for i in affected_indices:
            model_i = self.models[i]
            trainer = Trainer(model_i, lr=lr, weight_decay=wd, device=self.device)
            t0 = time.time()
            trainer.train(self.subgraphs[i], epochs=epochs, early_stop=early_stop)
            retrain_time += float(time.time() - t0)
            model_i.eval()
            with torch.no_grad():
                logits_i, hidden_i = model_i(
                    self.subgraphs[i].features.to(self.device),
                    self.subgraphs[i].edge_index.to(self.device),
                    return_hidden=True,
                )
            new_logits.append(logits_i)
            new_hidden.append(hidden_i)
        return new_logits, new_hidden, retrain_time, update_time

    def delete_edge(self, u: int, v: int, lr: float, wd: float, epochs: int, early_stop: int = 20) -> Tuple[List[torch.Tensor], List[torch.Tensor], float, float]:
        ei_all = self.g.edge_index
        global_mask = (((ei_all[0] == u) & (ei_all[1] == v)) | ((ei_all[0] == v) & (ei_all[1] == u)))
        global_idx = torch.nonzero(global_mask, as_tuple=False).view(-1)

        affected_indices = []
        if global_idx.numel() > 0:
            if self.assign.is_head_edge[global_idx].any().item():
                affected_indices = list(range(self.assign.num_subgraphs))
            else:
                subs = self.assign.tail_edge_subgraph[global_idx]
                subs = subs[subs >= 0]
                affected_indices = torch.unique(subs).tolist()

        if len(affected_indices) == 0:
            for i, sg in enumerate(self.subgraphs):
                ei = sg.edge_index
                has_uv = (((ei[0] == u) & (ei[1] == v)) | ((ei[0] == v) & (ei[1] == u))).any().item()
                if has_uv:
                    affected_indices.append(i)

        self.last_affected_indices = affected_indices
        if len(affected_indices) == 0:
            return [], [], 0.0, 0.0

        t_update = time.time()
        for i in affected_indices:
            self.subgraphs[i] = self._remove_edge_in_subgraph(self.subgraphs[i], u, v)
        update_time = float(time.time() - t_update)

        new_logits, new_hidden = [], []
        retrain_time = 0.0
        for i in affected_indices:
            model_i = self.models[i]
            trainer = Trainer(model_i, lr=lr, weight_decay=wd, device=self.device)
            t0 = time.time()
            trainer.train(self.subgraphs[i], epochs=epochs, early_stop=early_stop)
            retrain_time += float(time.time() - t0)
            model_i.eval()
            with torch.no_grad():
                logits_i, hidden_i = model_i(
                    self.subgraphs[i].features.to(self.device),
                    self.subgraphs[i].edge_index.to(self.device),
                    return_hidden=True,
                )
            new_logits.append(logits_i)
            new_hidden.append(hidden_i)
        return new_logits, new_hidden, retrain_time, update_time

    def recompute_fusion(self, lamda: float, head_mask: torch.Tensor) -> Tuple[torch.Tensor, float, List[torch.Tensor], torch.Tensor]:
        t0 = time.time()
        embed_list, pred_list = [], []
        for i, sg in enumerate(self.subgraphs):
            self.models[i].eval()
            with torch.no_grad():
                logits_i, hidden_i = self.models[i](
                    sg.features.to(self.device),
                    sg.edge_index.to(self.device),
                    return_hidden=True,
                )
                pred_list.append(F.softmax(logits_i, dim=1))
                embed_list.append(hidden_i)
        alpha = compute_struct_weights(embed_list, head_mask, lamda=lamda)
        fused_probs = fuse_predictions(pred_list, alpha)
        fusion_time = float(time.time() - t0)
        return fused_probs, fusion_time, pred_list, alpha
