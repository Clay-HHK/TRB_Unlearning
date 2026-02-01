import argparse
import time
import csv
from pathlib import Path
import numpy as np
import torch
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.graph_data import GraphData
from src.data.loader import load_from_src
from src.unlearning.retraining import RetrainConfig, retrain_after_delete
from src.unlearning.delete_only import DeleteSimilarityConfig, delete_similarity_train
from src.unlearning.influence import InfluenceConfig, influence_unlearn
from src.unlearning.bekm import BEKMConfig, BEKMUnlearningManager
from src.evaluation.long_tail import split_head_tail_by_degree
from src.attack.mia import membership_auc_by_groups, membership_auc, local_mask_radius1
from src.attack.mia import membership_auc_by_distance, membership_auc_dist_by_groups

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
    parser = argparse.ArgumentParser("Unlearning baselines")
    parser.add_argument("--dataset", required=True, choices=["DBLP_bipartite", "CiteSeer_bipartite", "ogbn-arxiv", "ogbn-products"])
    parser.add_argument("--source", choices=["npz", "src"], default="npz")
    parser.add_argument("--baseline", choices=["retraining", "delete", "if", "bekm"], required=True)
    parser.add_argument("--delete_node", type=int, default=None)
    parser.add_argument("--delete_edge", nargs=2, type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", choices=["gcn", "gat", "sage", "gin"], default="gcn")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--mlp", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=1e-3)
    parser.add_argument("--head_ratio", type=float, default=0.2)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--if_damping", type=float, default=1e-3)
    parser.add_argument("--if_scale", type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.source == "npz":
        # Adjust paths to rb_unlearning/data
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

    if args.dataset == "ogbn-products" and args.baseline in ("retraining", "delete", "if") and args.source == "npz":
        print("ogbn-products is large. Use --source=src and ensure memory is sufficient.")
        # return

    train_time = 0.0
    retrain_time = 0.0
    fusion_time = 0.0

    if args.baseline == "retraining":
        t0 = time.time()
        cfg = RetrainConfig(model=args.model, hidden=args.hidden, mlp_hidden=args.mlp, dropout=args.dropout, heads=args.heads, symmetrize_edges=True, lr=args.lr, wd=args.wd, epochs=args.epochs, device=args.device, head_ratio=args.head_ratio)
        res = retrain_after_delete(g, cfg, delete_node=args.delete_node, delete_edge=tuple(args.delete_edge) if args.delete_edge else None)
        retrain_time = time.time() - t0
    elif args.baseline == "delete":
        t0 = time.time()
        cfg = DeleteSimilarityConfig(model=args.model, hidden=args.hidden, mlp_hidden=args.mlp, dropout=args.dropout, heads=args.heads, symmetrize_edges=True, lr=args.lr, wd=args.wd, epochs=args.epochs, device=args.device, head_ratio=args.head_ratio, alpha=1.0)
        res = delete_similarity_train(g, cfg, delete_node=args.delete_node, delete_edge=tuple(args.delete_edge) if args.delete_edge else None)
        train_time = time.time() - t0
    elif args.baseline == "if":
        cfg = InfluenceConfig(model=args.model, hidden=args.hidden, mlp_hidden=args.mlp, dropout=args.dropout, heads=args.heads, symmetrize_edges=True, lr=args.lr, wd=args.wd, epochs=args.epochs, device=args.device, head_ratio=args.head_ratio, damping=args.if_damping, scale=args.if_scale)
        res = influence_unlearn(g, cfg, delete_node=args.delete_node, delete_edge=tuple(args.delete_edge) if args.delete_edge else None)
        train_time = float(res.get("TrainTime", 0.0))
        retrain_time = float(res.get("UpdateTime", 0.0))
    else:
        cfg = BEKMConfig(model=args.model, hidden=args.hidden, mlp_hidden=args.mlp, dropout=args.dropout, heads=args.heads, symmetrize_edges=True, lr=args.lr, wd=args.wd, epochs=args.epochs, device=args.device, K=args.K, seed=args.seed, head_ratio=args.head_ratio)
        t_init = time.time()
        mgr = BEKMUnlearningManager(g, cfg)
        mgr.train_initial()
        train_time = time.time() - t_init
        if args.delete_node is not None:
            retrain_time = float(mgr.delete_node(int(args.delete_node)) or 0.0)
        if args.delete_edge is not None:
            u, v = tuple(args.delete_edge)
            retrain_time = float(mgr.delete_edge(int(u), int(v)) or retrain_time)
        t_fuse = time.time()
        res = mgr.evaluate()
        fusion_time = float(res.get("FusionTime", time.time() - t_fuse))
        # base_probs = mgr.fused_probs().to(args.device)

    print(f"[{args.baseline}] ValAcc={res['ValAcc']:.4f}, TestAcc={res['TestAcc']:.4f}, ValF1={res['ValF1']:.4f}, TestF1={res['TestF1']:.4f}")
    print(f"[{args.baseline}] HeadAcc(Test)={res['TestHeadAcc']:.4f}, TailAcc(Test)={res['TestTailAcc']:.4f}")

    base_probs = None
    if args.baseline == "retraining":
        cfg0 = RetrainConfig(model=args.model, hidden=args.hidden, mlp_hidden=args.mlp, dropout=args.dropout, heads=args.heads, symmetrize_edges=True, lr=args.lr, wd=args.wd, epochs=args.epochs, device=args.device, head_ratio=args.head_ratio)
        res0 = retrain_after_delete(g, cfg0)
        base_probs = res0['probs'].to(args.device)
    elif args.baseline == "delete":
        cfg0 = DeleteSimilarityConfig(model=args.model, hidden=args.hidden, mlp_hidden=args.mlp, dropout=args.dropout, heads=args.heads, symmetrize_edges=True, lr=args.lr, wd=args.wd, epochs=args.epochs, device=args.device, head_ratio=args.head_ratio, alpha=1.0)
        res0 = delete_similarity_train(g, cfg0)
        base_probs = res0['probs'].to(args.device)
    elif args.baseline == "if":
        cfg0 = InfluenceConfig(model=args.model, hidden=args.hidden, mlp_hidden=args.mlp, dropout=args.dropout, heads=args.heads, symmetrize_edges=True, lr=args.lr, wd=args.wd, epochs=args.epochs, device=args.device, head_ratio=args.head_ratio, damping=args.if_damping, scale=args.if_scale)
        res0 = influence_unlearn(g, cfg0)
        base_probs = res0['probs'].to(args.device)

    split = split_head_tail_by_degree(g.edge_index, g.num_nodes, head_ratio=args.head_ratio)
    head_mask = split["head_mask"]
    mia = membership_auc_by_groups(
        res['probs'].to(args.device),
        g.train_mask.to(args.device),
        g.test_mask.to(args.device),
        head_mask.to(args.device),
        (~head_mask).to(args.device),
    )
    print(f"[MIA-Conf][{args.baseline}] AUC(All)={mia['All']:.4f}, Head={mia['Head']:.4f}, Tail={mia['Tail']:.4f}")

    if base_probs is not None:
        mia_dist = membership_auc_dist_by_groups(
            base_probs, res['probs'].to(args.device), g.train_mask.to(args.device), g.test_mask.to(args.device),
            head_mask.to(args.device), (~head_mask).to(args.device)
        )
        print(f"[MIA-Dist][{args.baseline}] AUC(All)={mia_dist['All']:.4f}, Head={mia_dist['Head']:.4f}, Tail={mia_dist['Tail']:.4f}")
    else:
        mia_dist = {"All": 0.0, "Head": 0.0, "Tail": 0.0}

    mia_local = None; mia_local_dist = None
    if args.delete_node is not None:
        centers = torch.tensor([int(args.delete_node)], device=args.device, dtype=torch.long)
        local = local_mask_radius1(g.edge_index.to(args.device), centers, g.num_nodes)
        mia_local = float(membership_auc(res['probs'].to(args.device), g.train_mask.to(args.device), g.test_mask.to(args.device), mask=local))
        if base_probs is not None:
            mia_local_dist = float(membership_auc_by_distance(base_probs, res['probs'].to(args.device), g.train_mask.to(args.device), g.test_mask.to(args.device), mask=local))
        print(f"[MIA-Conf][{args.baseline}] AUC(Local-Node-{int(args.delete_node)})={mia_local:.4f}")
        if mia_local_dist is not None:
            print(f"[MIA-Dist][{args.baseline}] AUC(Local-Node-{int(args.delete_node)})={mia_local_dist:.4f}")
    if args.delete_edge is not None:
        u, v = tuple(map(int, args.delete_edge))
        centers = torch.tensor([u, v], device=args.device, dtype=torch.long)
        local = local_mask_radius1(g.edge_index.to(args.device), centers, g.num_nodes)
        mia_local = float(membership_auc(res['probs'].to(args.device), g.train_mask.to(args.device), g.test_mask.to(args.device), mask=local))
        if base_probs is not None:
            mia_local_dist = float(membership_auc_by_distance(base_probs, res['probs'].to(args.device), g.train_mask.to(args.device), g.test_mask.to(args.device), mask=local))
        print(f"[MIA-Conf][{args.baseline}] AUC(Local-Edge-{u}-{v})={mia_local:.4f}")
        if mia_local_dist is not None:
            print(f"[MIA-Dist][{args.baseline}] AUC(Local-Edge-{u}-{v})={mia_local_dist:.4f}")

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "unlearning_baselines.csv"
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
                "dataset","source","baseline","model","hidden","heads","epochs","lr","wd","head_ratio","K","seed",
                "delete_type","delete_detail",
                "TrainTime","RetrainTime","FusionTime",
                "ValAcc","TestAcc","ValMicroF1","TestMicroF1","ValF1","TestF1","TestHeadAcc","TestTailAcc",
                "MIA_All","MIA_Head","MIA_Tail","MIA_Local",
                "MIA_Dist_All","MIA_Dist_Head","MIA_Dist_Tail","MIA_Dist_Local"
            ])
        w.writerow([
            args.dataset, args.source, args.baseline, args.model, args.hidden, args.heads, args.epochs, args.lr, args.wd, args.head_ratio, args.K, args.seed,
            delete_type, delete_detail,
            f"{train_time:.6f}", f"{retrain_time:.6f}", f"{fusion_time:.6f}",
            f"{res['ValAcc']:.6f}", f"{res['TestAcc']:.6f}",
            f"{res.get('ValMicroF1',0):.6f}", f"{res.get('TestMicroF1',0):.6f}",
            f"{res['ValF1']:.6f}", f"{res['TestF1']:.6f}", f"{res['TestHeadAcc']:.6f}", f"{res['TestTailAcc']:.6f}",
            f"{mia['All']:.6f}", f"{mia['Head']:.6f}", f"{mia['Tail']:.6f}", "" if mia_local is None else f"{mia_local:.6f}",
            f"{mia_dist['All']:.6f}", f"{mia_dist['Head']:.6f}", f"{mia_dist['Tail']:.6f}", "" if mia_local_dist is None else f"{mia_local_dist:.6f}"
        ])

if __name__ == "__main__":
    main()
