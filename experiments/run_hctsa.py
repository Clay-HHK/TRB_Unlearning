import argparse
import time
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.graph_data import GraphData
from src.data.loader import load_from_src
from src.training.trainer import Trainer
from src.models.gcn import SimpleGCN
from src.fusion.hctsa import (
    make_model,
    build_edge_subgraphs_with_assignments,
    build_edge_subgraphs_fast_cuda,
    build_edge_subgraphs_unbalanced_with_assignments,
    build_node_subgraphs_random_with_assignments,
)
from src.unlearning.hctsa_unlearning import HCTSAUnlearningManager
from src.utils.seed import set_seed
from src.utils.edge_order_cache import load_edge_order, save_edge_order, compute_edge_order
from src.evaluation.metrics import accuracy, macro_f1
from src.evaluation.long_tail import split_head_tail_by_degree, compute_group_metrics
from src.attack.mia import (
    membership_auc_by_groups,
    membership_auc,
    membership_auc_by_distance,
    membership_auc_dist_by_groups,
    local_mask_radius1,
    graybox_level2_mia_weight_aware_dist,
    graybox_level2_mia_partition_aware_dist,
    graybox_level3_mia_deletion_aware,
    whitebox_mia_gradient_based,
    whitebox_mia_parameter_based,
    graybox_mia_shard_analysis,
    graybox_mia_shard_distance
)

def load_graph_npz(path: Path) -> GraphData:
    d = np.load(path)
    return GraphData(
        features=torch.as_tensor(d["features"], dtype=torch.float32),
        labels=torch.as_tensor(d["labels"], dtype=torch.long).view(-1),
        edge_index=torch.as_tensor(d["edge_index"], dtype=torch.long),
        train_mask=torch.as_tensor(d["train_mask"].astype(bool)),
        val_mask=torch.as_tensor(d["val_mask"].astype(bool)),
        test_mask=torch.as_tensor(d["test_mask"].astype(bool)),
    )

def main():
    parser = argparse.ArgumentParser("HCTSA Unlearning Experiment")
    parser.add_argument("--dataset", required=True, choices=["DBLP_bipartite", "CiteSeer_bipartite", "ogbn-arxiv", "ogbn-products"])
    parser.add_argument("--source", choices=["npz", "src"], default="npz")
    parser.add_argument("--model", choices=["gcn", "gat", "sage", "gin"], default="gcn")
    parser.add_argument("--partition", choices=["hctsa", "balanced", "random"], default="hctsa")
    
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--head_ratio", type=float, default=0.2)
    parser.add_argument("--lamda", type=float, default=1.0)
    
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--mlp", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=1e-3)
    
    parser.add_argument("--delete_node", type=int, default=None)
    parser.add_argument("--delete_edge", nargs=2, type=int, default=None)
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_edge_order_cache", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.source == "npz":
        base_data_dir = Path(__file__).resolve().parent.parent / "data"
        if args.dataset == "DBLP_bipartite":
            path = base_data_dir / "DBLP/graph.npz"
        elif args.dataset == "ogbn-arxiv":
            path = base_data_dir / "ogbn-arxiv/graph.npz"
        else:
            path = base_data_dir / "ogbn-products/graph.npz"
        
        if path.exists():
            g = load_graph_npz(path)
        else:
            print(f"Warning: {path} not found, falling back to src loader")
            g = load_from_src(args.dataset, seed=args.seed)
    else:
        g = load_from_src(args.dataset, seed=args.seed)

    num_classes = int(torch.unique(g.labels[g.labels >= 0]).numel())
    in_dim = g.num_features

    # Edge order cache
    edge_order = None
    if args.use_edge_order_cache:
        cache_dir = Path(__file__).resolve().parent.parent / "data/cache"
        edge_order = load_edge_order(cache_dir, args.dataset, g.num_edges)
        if edge_order is None:
            print("Computing edge order...")
            edge_order = compute_edge_order(g, device=args.device)
            save_edge_order(cache_dir, args.dataset, edge_order)

    # Partitioning
    t_part = time.time()
    if args.partition == "hctsa":
        subgraphs, head_mask, E_head, assign = build_edge_subgraphs_fast_cuda(
            g, args.head_ratio, args.K, device=args.device, edge_order=edge_order
        )
    elif args.partition == "balanced":
        subgraphs, head_mask, E_head, assign = build_edge_subgraphs_unbalanced_with_assignments(
            g, args.head_ratio, args.K, edge_order=edge_order
        )
    else: # random
        subgraphs, head_mask, E_head, assign = build_node_subgraphs_random_with_assignments(
            g, args.K, args.head_ratio
        )
    partition_time = time.time() - t_part

    # Train Initial
    models = []
    t_train = time.time()
    for i in range(len(subgraphs)):
        model = make_model(args.model, in_dim, args.hidden, num_classes, args.dropout, args.mlp, args.heads, symmetrize_edges=True)
        trainer = Trainer(model, lr=args.lr, weight_decay=args.wd, device=args.device)
        trainer.train(subgraphs[i], epochs=args.epochs, early_stop=20)
        models.append(model)
    train_time = time.time() - t_train

    # Manager
    mgr = HCTSAUnlearningManager(g, subgraphs, assign, models, args.device, num_classes)

    # Initial Evaluate
    probs_before, fusion_time, probs_shard_before, alpha_before = mgr.recompute_fusion(args.lamda, head_mask)
    
    # Deletion
    retrain_time = 0.0
    update_time = 0.0
    new_logits = [] # Shard logits after update
    
    if args.delete_node is not None:
        new_logits, _, retrain_time, update_time = mgr.delete_node(int(args.delete_node), args.lr, args.wd, max(50, args.epochs // 2))
    elif args.delete_edge is not None:
        u, v = tuple(map(int, args.delete_edge))
        new_logits, _, retrain_time, update_time = mgr.delete_edge(u, v, args.lr, args.wd, max(50, args.epochs // 2))
    
    # Final Evaluate
    probs_after, fusion_time2, probs_shard_after, alpha_after = mgr.recompute_fusion(args.lamda, head_mask)
    fusion_time += fusion_time2

    y = g.labels.to(args.device)
    val_mask = g.val_mask.to(args.device)
    test_mask = g.test_mask.to(args.device)
    
    val_acc = accuracy(probs_after, y, val_mask)
    test_acc = accuracy(probs_after, y, test_mask)
    val_f1 = macro_f1(probs_after, y, val_mask)
    test_f1 = macro_f1(probs_after, y, test_mask)
    
    split = split_head_tail_by_degree(g.edge_index, g.num_nodes, head_ratio=args.head_ratio)
    val_group = compute_group_metrics(probs_after, y, val_mask, split)
    test_group = compute_group_metrics(probs_after, y, test_mask, split)

    print(f"[HCTSA] ValAcc={val_acc:.4f}, TestAcc={test_acc:.4f}, ValF1={val_f1:.4f}, TestF1={test_f1:.4f}")

    # MIA Evaluation
    mia = membership_auc_dist_by_groups(
        probs_before, probs_after, g.train_mask.to(args.device), g.test_mask.to(args.device),
        head_mask.to(args.device), (~head_mask).to(args.device)
    )
    print(f"[MIA-Dist] All={mia['All']:.4f}, Head={mia['Head']:.4f}, Tail={mia['Tail']:.4f}")

    # Save results
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "hctsa_results.csv"
    need_header = (not out_csv.exists())
    
    delete_type = "none"
    delete_detail = "NA"
    if args.delete_node is not None:
        delete_type = "node"; delete_detail = str(int(args.delete_node))
    elif args.delete_edge is not None:
        u, v = tuple(map(int, args.delete_edge)); delete_type = "edge"; delete_detail = f"{u}-{v}"

    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow([
                "dataset","partition","K","head_ratio","lamda","model","hidden","epochs","lr","wd","seed",
                "delete_type","delete_detail",
                "PartitionTime","TrainTime","RetrainTime","UpdateTime","FusionTime",
                "ValAcc","TestAcc","ValF1","TestF1",
                "TestHeadAcc","TestTailAcc",
                "MIA_All","MIA_Head","MIA_Tail"
            ])
        w.writerow([
            args.dataset, args.partition, args.K, args.head_ratio, args.lamda, args.model, args.hidden, args.epochs, args.lr, args.wd, args.seed,
            delete_type, delete_detail,
            f"{partition_time:.6f}", f"{train_time:.6f}", f"{retrain_time:.6f}", f"{update_time:.6f}", f"{fusion_time:.6f}",
            f"{val_acc:.6f}", f"{test_acc:.6f}", f"{val_f1:.6f}", f"{test_f1:.6f}",
            f"{test_group['HeadAcc']:.6f}", f"{test_group['TailAcc']:.6f}",
            f"{mia['All']:.6f}", f"{mia['Head']:.6f}", f"{mia['Tail']:.6f}"
        ])

if __name__ == "__main__":
    main()
