import torch
from typing import Dict, Optional, List, Any

def _roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.view(-1).detach()
    labels = labels.view(-1).detach()
    dev = scores.device
    order = torch.argsort(scores)
    ranks = torch.zeros_like(order, dtype=torch.float32, device=dev)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32, device=dev)
    pos = (labels == 1)
    neg = (labels == 0)
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[pos].sum().item()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def membership_auc_by_distance(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    scores = torch.norm(probs_a - probs_b, dim=1)
    if mask is not None:
        scores = scores[mask]
        train_mask = train_mask[mask]
        test_mask = test_mask[mask]
    labels = torch.zeros_like(scores, dtype=torch.float32)
    labels[train_mask] = 1.0
    labels[~train_mask & test_mask] = 0.0
    sel = train_mask | test_mask
    if int(sel.sum().item()) == 0:
        return float("nan")
    return _roc_auc(scores[sel], labels[sel])

def membership_auc_from_scores(
    scores: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    if not isinstance(scores, torch.Tensor):
        if callable(scores):
            try:
                scores = scores()
            except:
                pass
        if not isinstance(scores, torch.Tensor):
            print(f"[Error] membership_auc_from_scores received {type(scores)} instead of Tensor.")
            return float("nan")

    if mask is not None:
        scores = scores[mask]
        train_mask = train_mask[mask]
        test_mask = test_mask[mask]
    labels = torch.zeros_like(scores, dtype=torch.float32)
    labels[train_mask] = 1.0
    labels[~train_mask & test_mask] = 0.0
    sel = train_mask | test_mask
    if int(sel.sum().item()) == 0:
        return float("nan")
    return _roc_auc(scores[sel], labels[sel])

def membership_auc(
    probs: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    scores = probs.max(dim=1).values
    return membership_auc_from_scores(scores, train_mask, test_mask, mask)

def graybox_level2_mia_weight_aware_dist(
    probs_per_shard_before: List[torch.Tensor],
    probs_per_shard_after: List[torch.Tensor],
    alpha_before: torch.Tensor,
    alpha_after: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    head_mask: torch.Tensor,
    tail_mask: torch.Tensor,
) -> Dict[str, float]:
    pb = torch.stack(probs_per_shard_before, dim=0)
    pa = torch.stack(probs_per_shard_after, dim=0)
    ab = alpha_before.to(pb.device).view(-1, 1, 1)
    aa = alpha_after.to(pb.device).view(-1, 1, 1)
    wpb = (pb * ab).sum(dim=0)
    wpa = (pa * aa).sum(dim=0)
    dist = torch.norm(wpb - wpa, dim=1)
    results = {}
    results["Weight_Dist_All"] = membership_auc_from_scores(dist, train_mask, test_mask)
    results["Weight_Dist_Tail"] = membership_auc_from_scores(dist, train_mask, test_mask, mask=tail_mask)
    return results

def graybox_level2_mia_partition_aware_dist(
    probs_per_shard_before: List[torch.Tensor],
    probs_per_shard_after: List[torch.Tensor],
    node_aff: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    head_mask: torch.Tensor,
    tail_mask: torch.Tensor,
) -> Dict[str, float]:
    pb = torch.stack(probs_per_shard_before, dim=0)
    pa = torch.stack(probs_per_shard_after, dim=0)
    m = node_aff.t().to(pb.device).float().unsqueeze(-1)
    sum_b = (pb * m).sum(dim=0)
    sum_a = (pa * m).sum(dim=0)
    cnt = m.sum(dim=0) + 1e-8
    mpb = sum_b / cnt
    mpa = sum_a / cnt
    dist = torch.norm(mpb - mpa, dim=1)
    results = {}
    results["Partition_Dist_All"] = membership_auc_from_scores(dist, train_mask, test_mask)
    results["Partition_Dist_Tail"] = membership_auc_from_scores(dist, train_mask, test_mask, mask=tail_mask)
    return results

def graybox_level3_mia_deletion_aware(
    probs_per_shard_before: List[torch.Tensor],
    probs_per_shard_after: List[torch.Tensor],
    affected_shards: List[int],
    neighbor_mask: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> Dict[str, float]:
    results = {}
    if not affected_shards:
        return {"Deletion_Neighbor_Shard_Dist": float("nan")}

    dist_list = []
    for k in affected_shards:
        if k < len(probs_per_shard_before) and k < len(probs_per_shard_after):
            d = torch.norm(probs_per_shard_before[k] - probs_per_shard_after[k], dim=1)
            dist_list.append(d)
    
    if dist_list:
        max_dist = torch.stack(dist_list, dim=0).max(dim=0).values
        auc = membership_auc_from_scores(max_dist, train_mask, test_mask, mask=neighbor_mask)
        results["Deletion_Neighbor_Shard_Dist"] = auc
    else:
        results["Deletion_Neighbor_Shard_Dist"] = float("nan")

    return results

def whitebox_mia_gradient_based(
    models: List[torch.nn.Module],
    subgraphs: List[Any], 
    device: str,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    head_mask: torch.Tensor,
    tail_mask: torch.Tensor,
) -> Dict[str, float]:
    import torch.nn.functional as F
    grad_norms_list = []
    
    for sg, model in zip(subgraphs, models):
        model.eval()
        if sg.features.size(0) == 0:
             grad_norms_list.append(torch.zeros(train_mask.size(0), device=device))
             continue

        x = sg.features.to(device).clone().detach().requires_grad_(True)
        ei = sg.edge_index.to(device)
        y = sg.labels.to(device)
        
        logits = model(x, ei)
        
        mask_valid = (y >= 0)
        if mask_valid.sum() > 0:
            loss = F.cross_entropy(logits[mask_valid], y[mask_valid])
            model.zero_grad()
            loss.backward()
            grad_norm = x.grad.norm(dim=1).detach()
        else:
            grad_norm = torch.zeros(x.size(0), device=device)

        grad_norms_list.append(grad_norm)
        
    if not grad_norms_list:
        return {"Gradient_Input_Norm_All": float("nan"), "Gradient_Input_Norm_Tail": float("nan")}

    grad_norms = torch.stack(grad_norms_list, dim=0)
    mean_grad = grad_norms.mean(dim=0)
    score = -mean_grad
    
    results = {}
    results["Gradient_Input_Norm_All"] = membership_auc_from_scores(score, train_mask, test_mask)
    results["Gradient_Input_Norm_Tail"] = membership_auc_from_scores(score, train_mask, test_mask, mask=tail_mask)
    
    return results

def whitebox_mia_parameter_based(
    models_before: List[torch.nn.Module],
    models_after: List[torch.nn.Module],
) -> Dict[str, float]:
    total_dist = 0.0
    count = 0
    for m1, m2 in zip(models_before, models_after):
        dist = 0.0
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            dist += torch.norm(p1 - p2).item() ** 2
        total_dist += dist
        count += 1
    
    avg_dist = total_dist / max(1, count)
    return {"Parameter_L2_Distance": avg_dist}

def membership_auc_by_groups(
    probs: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    head_mask: torch.Tensor,
    tail_mask: torch.Tensor,
) -> Dict[str, float]:
    scores = probs.max(dim=1).values
    auc_all = membership_auc_from_scores(scores, train_mask, test_mask)
    auc_head = membership_auc_from_scores(scores, train_mask, test_mask, mask=head_mask)
    auc_tail = membership_auc_from_scores(scores, train_mask, test_mask, mask=tail_mask)
    return {"All": auc_all, "Head": auc_head, "Tail": auc_tail}

def membership_auc_dist_by_groups(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    head_mask: torch.Tensor,
    tail_mask: torch.Tensor,
) -> Dict[str, float]:
    auc_all = membership_auc_by_distance(probs_a, probs_b, train_mask, test_mask)
    auc_head = membership_auc_by_distance(probs_a, probs_b, train_mask, test_mask, mask=head_mask)
    auc_tail = membership_auc_by_distance(probs_a, probs_b, train_mask, test_mask, mask=tail_mask)
    return {"All": auc_all, "Head": auc_head, "Tail": auc_tail}

def local_mask_radius1(edge_index: torch.Tensor, centers: torch.Tensor, num_nodes: int) -> torch.Tensor:
    centers = centers.view(-1)
    src, dst = edge_index[0], edge_index[1]
    is_center = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
    is_center[centers] = True
    mask_src = is_center[src]
    mask_dst = is_center[dst]
    nb_from_src = dst[mask_src]
    nb_from_dst = src[mask_dst]
    neighbors = torch.unique(torch.cat([centers, nb_from_src, nb_from_dst], dim=0))
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
    mask[neighbors] = True
    return mask

def graybox_mia_shard_analysis(
    probs_per_shard: List[torch.Tensor],
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    head_mask: torch.Tensor,
    tail_mask: torch.Tensor,
) -> Dict[str, float]:
    confs = []
    for p in probs_per_shard:
        confs.append(p.max(dim=1).values)
    confs = torch.stack(confs, dim=0)
    
    max_conf = confs.max(dim=0).values
    mean_conf = confs.mean(dim=0)

    results = {}
    
    auc_all = membership_auc_from_scores(max_conf, train_mask, test_mask)
    auc_tail = membership_auc_from_scores(max_conf, train_mask, test_mask, mask=tail_mask)
    results["Shard_Max_Conf_All"] = auc_all
    results["Shard_Max_Conf_Tail"] = auc_tail

    auc_all = membership_auc_from_scores(mean_conf, train_mask, test_mask)
    auc_tail = membership_auc_from_scores(mean_conf, train_mask, test_mask, mask=tail_mask)
    results["Shard_Mean_Conf_All"] = auc_all
    results["Shard_Mean_Conf_Tail"] = auc_tail
    
    return results

def graybox_mia_shard_distance(
    probs_per_shard_before: List[torch.Tensor],
    probs_per_shard_after: List[torch.Tensor],
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    head_mask: torch.Tensor,
    tail_mask: torch.Tensor,
) -> Dict[str, float]:
    dists = []
    for pb, pa in zip(probs_per_shard_before, probs_per_shard_after):
        dists.append(torch.norm(pb - pa, dim=1))
    dists = torch.stack(dists, dim=0)
    max_dist = dists.max(dim=0).values
    mean_dist = dists.mean(dim=0)
    results = {}
    results["Shard_Max_Dist_All"] = membership_auc_from_scores(max_dist, train_mask, test_mask)
    results["Shard_Max_Dist_Tail"] = membership_auc_from_scores(max_dist, train_mask, test_mask, mask=tail_mask)
    results["Shard_Mean_Dist_All"] = membership_auc_from_scores(mean_dist, train_mask, test_mask)
    results["Shard_Mean_Dist_Tail"] = membership_auc_from_scores(mean_dist, train_mask, test_mask, mask=tail_mask)
    
    return results
