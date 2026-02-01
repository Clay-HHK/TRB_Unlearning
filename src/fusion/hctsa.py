import torch
import torch.nn.functional as F
import time
from typing import List, Tuple, Optional  # Added Optional here
from src.data.graph_data import GraphData
from src.training.trainer import Trainer
from src.models.gcn import SimpleGCN
from src.models.gat import SimpleGAT
from src.models.sage import GraphSAGE
from src.models.gin import GIN
from src.models.mlp import MLPClassifier
from dataclasses import dataclass

def build_subgraph(g: GraphData, node_mask: torch.Tensor) -> Tuple[GraphData, torch.Tensor]:
    """
    Induces a subgraph from the original graph based on a node mask.
    
    Args:
        g (GraphData): Original graph.
        node_mask (torch.Tensor): Boolean mask indicating nodes to keep.
        
    Returns:
        Tuple[GraphData, torch.Tensor]: The subgraph and the indices of kept nodes in the original graph.
    """
    node_mask = node_mask.view(-1)
    idx = torch.nonzero(node_mask, as_tuple=False).view(-1)
    # Mapping: original index -> subgraph index
    mapping = torch.full((g.num_nodes,), -1, dtype=torch.long, device=g.features.device)
    mapping[idx] = torch.arange(idx.size(0), device=g.features.device, dtype=torch.long)
    
    # Filter and remap edges
    ei = g.edge_index
    keep = node_mask[ei[0]] & node_mask[ei[1]]
    src = mapping[ei[0, keep]]
    dst = mapping[ei[1, keep]]
    new_ei = torch.stack([src, dst], dim=0)
    
    sub = GraphData(
        features=g.features[idx],
        labels=g.labels[idx],
        edge_index=new_ei,
        train_mask=g.train_mask[idx],
        val_mask=g.val_mask[idx],
        test_mask=g.test_mask[idx],
    )
    return sub, idx

def make_model(name: str, in_dim: int, hidden: int, out_dim: int, dropout: float, mlp_hidden: int, heads: int, symmetrize_edges: bool):
    """Factory function to create GNN models."""
    if name == "gcn":
        return SimpleGCN(in_dim=in_dim, hidden_dim=hidden, out_dim=out_dim, dropout=dropout, mlp_hidden=mlp_hidden, symmetrize_edges=symmetrize_edges)
    elif name == "gat":
        return SimpleGAT(in_dim=in_dim, hidden_dim=hidden, out_dim=out_dim, heads=heads, dropout=dropout, mlp_hidden=mlp_hidden, symmetrize_edges=symmetrize_edges)
    elif name == "sage":
        return GraphSAGE(in_dim=in_dim, hidden_dim=hidden, out_dim=out_dim, dropout=dropout, mlp_hidden=mlp_hidden, symmetrize_edges=symmetrize_edges)
    elif name == "gin":
        return GIN(in_dim=in_dim, hidden_dim=hidden, out_dim=out_dim, dropout=dropout, mlp_hidden=mlp_hidden, symmetrize_edges=symmetrize_edges)
    else:
        raise ValueError(f"unsupported model: {name}")

def analyze_head_tail_edges_by_score(g: GraphData, head_ratio: float, edge_order: torch.Tensor | None = None):
    ei = g.edge_index
    num_nodes = g.num_nodes
    nodes = torch.cat([ei[0], ei[1]], dim=0)
    deg = torch.bincount(nodes, minlength=num_nodes)
    src, dst = ei[0], ei[1]
    E = int(ei.size(1))
    k = max(1, int(E * head_ratio))
    if edge_order is None:
        scores = deg[src] + deg[dst]
        order = torch.argsort(scores, descending=True)
    else:
        order = edge_order.to(ei.device)
        if int(order.numel()) != E:
            scores = deg[src] + deg[dst]
            order = torch.argsort(scores, descending=True)
    head_edge_idx = order[:k]
    mask = torch.zeros(E, dtype=torch.bool, device=ei.device)
    mask[head_edge_idx] = True
    head_ei = ei[:, mask]
    tail_ei = ei[:, ~mask]
    
    node_order = torch.argsort(deg, descending=True)
    k_nodes = max(1, int(num_nodes * head_ratio))
    head_nodes = node_order[:k_nodes]
    head_mask = torch.zeros(num_nodes, dtype=torch.bool, device=ei.device)
    head_mask[head_nodes] = True
    return head_mask, head_ei, tail_ei

def build_edge_subgraphs(g: GraphData, head_ratio: float, num_subgraphs: int):
    head_mask, head_ei, tail_ei = analyze_head_tail_edges_by_score(g, head_ratio)
    subgraphs: List[GraphData] = []
    E_head = int(head_ei.size(1))
    E_tail = int(tail_ei.size(1))
    if E_tail == 0:
        for _ in range(num_subgraphs):
            subgraphs.append(GraphData(
                features=g.features, labels=g.labels, edge_index=head_ei,
                train_mask=g.train_mask, val_mask=g.val_mask, test_mask=g.test_mask,
            ))
        return subgraphs, head_mask, E_head
    
    idx = torch.arange(E_tail, device=tail_ei.device)
    for i in range(num_subgraphs):
        part = idx[i::num_subgraphs]
        sub_tail = tail_ei[:, part]
        new_ei = torch.cat([head_ei, sub_tail], dim=1)
        subgraphs.append(GraphData(
            features=g.features, labels=g.labels, edge_index=new_ei,
            train_mask=g.train_mask, val_mask=g.val_mask, test_mask=g.test_mask,
        ))
    return subgraphs, head_mask, E_head

def compute_struct_weights(embed_list: List[torch.Tensor], head_mask: torch.Tensor, lamda: float) -> torch.Tensor:
    device = embed_list[0].device
    head_mask = head_mask.to(device)
    head_idx = torch.nonzero(head_mask, as_tuple=False).view(-1)
    if head_idx.numel() == 0:
        return torch.ones(len(embed_list), device=device) / len(embed_list)
    sub_means = []
    for emb in embed_list:
        sub_means.append(emb[head_idx].mean(dim=0))
    global_mean = torch.stack(sub_means, dim=0).mean(dim=0)
    sims = []
    for sub_mean in sub_means:
        sim = F.cosine_similarity(sub_mean.unsqueeze(0), global_mean.unsqueeze(0)).squeeze(0)
        sims.append(sim)
    sims = torch.stack(sims)
    alpha = F.softmax(lamda * sims, dim=0)
    return alpha

def fuse_predictions_weighted(pred_list: List[torch.Tensor], alpha: torch.Tensor) -> torch.Tensor:
    fused = torch.zeros_like(pred_list[0])
    for w, p in zip(alpha, pred_list):
        fused += w * p
    return fused

def fuse_predictions(pred_list: List[torch.Tensor], alpha: torch.Tensor) -> torch.Tensor:
    return fuse_predictions_weighted(pred_list, alpha)

def fuse_predictions_vote(pred_list: List[torch.Tensor]) -> torch.Tensor:
    device = pred_list[0].device
    num_classes = pred_list[0].size(1)
    preds = torch.stack([p.argmax(dim=1) for p in pred_list], dim=0)
    mode_vals, _ = torch.mode(preds, dim=0)
    fused = torch.zeros(pred_list[0].size(0), num_classes, device=device)
    fused[torch.arange(fused.size(0), device=device), mode_vals] = 1.0
    return fused

def train_on_subgraph(
    subgraph: GraphData,
    model_name: str,
    hidden: int,
    mlp_hidden: int,
    dropout: float,
    heads: int,
    symmetrize_edges: bool,
    lr: float,
    wd: float,
    epochs: int,
    device: str,
    num_classes: int,
):
    in_dim = int(subgraph.features.size(1))
    model = make_model(model_name, in_dim, hidden, num_classes, dropout, mlp_hidden, heads, symmetrize_edges)
    trainer = Trainer(model, lr=lr, weight_decay=wd, device=device)
    trainer.train(subgraph, epochs=epochs, early_stop=20)
    model.eval()
    with torch.no_grad():
        logits, hidden_out = model(
            subgraph.features.to(trainer.device),
            subgraph.edge_index.to(trainer.device),
            return_hidden=True,
        )
    return model, logits, hidden_out


@dataclass
class SubgraphAssignments:
    is_head_edge: torch.Tensor
    tail_edge_subgraph: torch.Tensor
    node_aff_subgraphs: torch.Tensor
    num_subgraphs: int

def build_edge_subgraphs_with_assignments(g: GraphData, head_ratio: float, num_subgraphs: int, edge_order: torch.Tensor | None = None) -> Tuple[List[GraphData], torch.Tensor, int, SubgraphAssignments]:
    ei = g.edge_index
    num_nodes = g.num_nodes
    nodes = torch.cat([ei[0], ei[1]], dim=0)
    deg = torch.bincount(nodes, minlength=num_nodes)
    src, dst = ei[0], ei[1]
    E = int(ei.size(1))

    k_edges = max(1, int(E * head_ratio))
    if edge_order is None:
        scores = deg[src] + deg[dst]
        order = torch.argsort(scores, descending=True)
    else:
        order = edge_order.to(ei.device)
        if int(order.numel()) != E:
            scores = deg[src] + deg[dst]
            order = torch.argsort(scores, descending=True)
    head_edge_idx = order[:k_edges]
    is_head = torch.zeros(E, dtype=torch.bool, device=ei.device)
    is_head[head_edge_idx] = True

    tail_idx_all = torch.nonzero(~is_head, as_tuple=False).view(-1)
    head_ei = ei[:, is_head]
    tail_ei = ei[:, ~is_head]
    E_head = int(head_ei.size(1))
    E_tail = int(tail_ei.size(1))

    subgraphs: List[GraphData] = []
    tail_edge_subgraph = torch.full((E,), -1, dtype=torch.long, device=ei.device)
    node_aff = torch.zeros(num_nodes, num_subgraphs, dtype=torch.bool, device=ei.device)

    if E_tail == 0:
        for _ in range(num_subgraphs):
            subgraphs.append(GraphData(
                features=g.features, labels=g.labels, edge_index=head_ei,
                train_mask=g.train_mask, val_mask=g.val_mask, test_mask=g.test_mask,
            ))
    else:
        perm = torch.randperm(E_tail, device=ei.device)
        bins = perm % num_subgraphs
        for i in range(num_subgraphs):
            part_local = perm[bins == i]
            sub_tail = tail_ei[:, part_local]
            new_ei = torch.cat([head_ei, sub_tail], dim=1)

            subgraphs.append(GraphData(
                features=g.features, labels=g.labels, edge_index=new_ei,
                train_mask=g.train_mask, val_mask=g.val_mask, test_mask=g.test_mask,
            ))

            part_global = tail_idx_all[part_local]
            tail_edge_subgraph[part_global] = i

            if sub_tail.numel() > 0:
                uniq_nodes = torch.unique(torch.cat([sub_tail[0], sub_tail[1]], dim=0))
                node_aff[uniq_nodes, i] = True

    if head_ei.numel() > 0:
        h_src, h_dst = head_ei[0], head_ei[1]
        node_aff[h_src, :] = True
        node_aff[h_dst, :] = True

    assignments = SubgraphAssignments(is_head, tail_edge_subgraph, node_aff, num_subgraphs)

    node_order = torch.argsort(deg, descending=True)
    k_nodes = max(1, int(num_nodes * head_ratio))
    head_nodes = node_order[:k_nodes]
    head_mask = torch.zeros(num_nodes, dtype=torch.bool, device=ei.device)
    head_mask[head_nodes] = True
    return subgraphs, head_mask, E_head, assignments

def build_edge_subgraphs_fast_cuda(g: GraphData, head_ratio: float, num_subgraphs: int, device: str | None = None, edge_order: torch.Tensor | None = None) -> Tuple[List[GraphData], torch.Tensor, int, SubgraphAssignments]:
    ei = g.edge_index.to(device) if device is not None else g.edge_index
    num_nodes = g.num_nodes
    nodes = torch.cat([ei[0], ei[1]], dim=0)
    deg = torch.bincount(nodes, minlength=num_nodes)
    src, dst = ei[0], ei[1]
    E = int(ei.size(1))
    k_edges = max(1, int(E * head_ratio))
    if edge_order is None:
        scores = deg[src] + deg[dst]
        order = torch.argsort(scores, descending=True)
    else:
        order = edge_order.to(ei.device)
        if int(order.numel()) != E:
            scores = deg[src] + deg[dst]
            order = torch.argsort(scores, descending=True)
    head_edge_idx = order[:k_edges]
    is_head = torch.zeros(E, dtype=torch.bool, device=ei.device)
    is_head[head_edge_idx] = True

    tail_idx_all = torch.nonzero(~is_head, as_tuple=False).view(-1)
    head_ei = ei[:, is_head]
    tail_ei = ei[:, ~is_head]
    E_head = int(head_ei.size(1))
    E_tail = int(tail_ei.size(1))

    subgraphs: List[GraphData] = []
    tail_edge_subgraph = torch.full((E,), -1, dtype=torch.long, device=ei.device)
    node_aff = torch.zeros(num_nodes, num_subgraphs, dtype=torch.bool, device=ei.device)

    if E_tail == 0:
        for _ in range(num_subgraphs):
            subgraphs.append(GraphData(
                features=g.features, labels=g.labels, edge_index=head_ei,
                train_mask=g.train_mask, val_mask=g.val_mask, test_mask=g.test_mask,
            ))
    else:
        perm = torch.randperm(E_tail, device=ei.device)
        bins = perm % num_subgraphs
        for i in range(num_subgraphs):
            part_local = perm[bins == i]
            sub_tail = tail_ei[:, part_local]
            new_ei = torch.cat([head_ei, sub_tail], dim=1)

            subgraphs.append(GraphData(
                features=g.features, labels=g.labels, edge_index=new_ei,
                train_mask=g.train_mask, val_mask=g.val_mask, test_mask=g.test_mask,
            ))

            part_global = tail_idx_all[part_local]
            tail_edge_subgraph[part_global] = i

            if sub_tail.numel() > 0:
                uniq_nodes = torch.unique(torch.cat([sub_tail[0], sub_tail[1]], dim=0))
                node_aff[uniq_nodes, i] = True

    if head_ei.numel() > 0:
        h_src, h_dst = head_ei[0], head_ei[1]
        node_aff[h_src, :] = True
        node_aff[h_dst, :] = True

    assignments = SubgraphAssignments(is_head, tail_edge_subgraph, node_aff, num_subgraphs)

    node_order = torch.argsort(deg, descending=True)
    k_nodes = max(1, int(num_nodes * head_ratio))
    head_nodes = node_order[:k_nodes]
    head_mask = torch.zeros(num_nodes, dtype=torch.bool, device=ei.device)
    head_mask[head_nodes] = True
    return subgraphs, head_mask, E_head, assignments

def build_edge_subgraphs_unbalanced_with_assignments(
    g: GraphData,
    head_ratio: float,
    num_subgraphs: int,
    edge_order: torch.Tensor | None = None,
) -> Tuple[List[GraphData], torch.Tensor, int, SubgraphAssignments]:
    ei = g.edge_index
    num_nodes = g.num_nodes
    nodes = torch.cat([ei[0], ei[1]], dim=0)
    deg = torch.bincount(nodes, minlength=num_nodes)
    src, dst = ei[0], ei[1]
    E = int(ei.size(1))
    k_edges = max(1, int(E * head_ratio))
    if edge_order is None:
        scores = deg[src] + deg[dst]
        order = torch.argsort(scores, descending=True)
    else:
        order = edge_order.to(ei.device)
        if int(order.numel()) != E:
            scores = deg[src] + deg[dst]
            order = torch.argsort(scores, descending=True)
    head_edge_idx = order[:k_edges]
    is_head = torch.zeros(E, dtype=torch.bool, device=ei.device)
    is_head[head_edge_idx] = True

    tail_idx_all = torch.nonzero(~is_head, as_tuple=False).view(-1)
    head_ei = ei[:, is_head]
    tail_ei = ei[:, ~is_head]
    E_head = int(head_ei.size(1))
    E_tail = int(tail_ei.size(1))

    subgraphs: List[GraphData] = []
    tail_edge_subgraph = torch.full((E,), -1, dtype=torch.long, device=ei.device)
    node_aff = torch.zeros(num_nodes, num_subgraphs, dtype=torch.bool, device=ei.device)

    if E_tail == 0:
        for _ in range(num_subgraphs):
            subgraphs.append(GraphData(
                features=g.features, labels=g.labels, edge_index=head_ei,
                train_mask=g.train_mask, val_mask=g.val_mask, test_mask=g.test_mask,
            ))
    else:
        bins = torch.randint(low=0, high=num_subgraphs, size=(E_tail,), device=ei.device)
        for i in range(num_subgraphs):
            part_local = torch.nonzero(bins == i, as_tuple=False).view(-1)
            if part_local.numel() > 0:
                sub_tail = tail_ei[:, part_local]
                new_ei = torch.cat([head_ei, sub_tail], dim=1)
                part_global = tail_idx_all[part_local]
                tail_edge_subgraph[part_global] = i
                uniq_nodes = torch.unique(torch.cat([sub_tail[0], sub_tail[1]], dim=0))
                node_aff[uniq_nodes, i] = True
            else:
                new_ei = head_ei
            subgraphs.append(GraphData(
                features=g.features, labels=g.labels, edge_index=new_ei,
                train_mask=g.train_mask, val_mask=g.val_mask, test_mask=g.test_mask,
            ))

    if head_ei.numel() > 0:
        h_src, h_dst = head_ei[0], head_ei[1]
        node_aff[h_src, :] = True
        node_aff[h_dst, :] = True

    assignments = SubgraphAssignments(is_head, tail_edge_subgraph, node_aff, num_subgraphs)

    node_order = torch.argsort(deg, descending=True)
    k_nodes = max(1, int(num_nodes * head_ratio))
    head_nodes = node_order[:k_nodes]
    head_mask = torch.zeros(num_nodes, dtype=torch.bool, device=ei.device)
    head_mask[head_nodes] = True
    return subgraphs, head_mask, E_head, assignments

def build_node_subgraphs_random_with_assignments(
    g: GraphData,
    num_subgraphs: int,
    head_ratio: float,
) -> Tuple[List[GraphData], torch.Tensor, int, SubgraphAssignments]:
    N = g.num_nodes
    device = g.edge_index.device
    perm = torch.randperm(N, device=device)
    bins = perm % num_subgraphs

    subgraphs: List[GraphData] = []
    E = int(g.edge_index.size(1))
    is_head = torch.zeros(E, dtype=torch.bool, device=device)
    tail_edge_subgraph = torch.full((E,), -1, dtype=torch.long, device=device)
    node_aff = torch.zeros(N, num_subgraphs, dtype=torch.bool, device=device)

    head_mask, _, _ = analyze_head_tail_edges_by_score(g, head_ratio)

    ei = g.edge_index
    src_all, dst_all = ei[0], ei[1]
    for i in range(num_subgraphs):
        nodes_i = perm[bins == i]
        node_mask_i = torch.zeros(N, dtype=torch.bool, device=device)
        node_mask_i[nodes_i] = True

        mask_ei_i = node_mask_i[src_all] & node_mask_i[dst_all]
        new_ei = ei[:, mask_ei_i]

        idx_global = torch.nonzero(mask_ei_i, as_tuple=False).view(-1)
        tail_edge_subgraph[idx_global] = i

        node_aff[nodes_i, i] = True

        subgraphs.append(GraphData(
            features=g.features,
            labels=g.labels,
            edge_index=new_ei,
            train_mask=(g.train_mask & node_mask_i),
            val_mask=(g.val_mask & node_mask_i),
            test_mask=(g.test_mask & node_mask_i),
        ))

    assignments = SubgraphAssignments(is_head, tail_edge_subgraph, node_aff, num_subgraphs)
    num_head_edges = 0
    return subgraphs, head_mask, num_head_edges, assignments
