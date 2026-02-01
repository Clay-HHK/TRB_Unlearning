import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.graph_data import GraphData
from src.data.loader import load_from_src
from src.fusion.hctsa import make_model, build_subgraph
from src.training.trainer import Trainer
from src.utils.seed import set_seed
from src.evaluation.metrics import accuracy, macro_f1, micro_f1
from src.evaluation.long_tail import split_head_tail_by_degree, compute_group_metrics

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

def _train_one_subgraph(subgraph: GraphData, model_name: str, in_dim: int, hidden: int, out_dim: int,
                        dropout: float, mlp_hidden: int, heads: int, symmetrize_edges: bool,
                        lr: float, wd: float, epochs: int, device: str):
    model = make_model(model_name, in_dim, hidden, out_dim, dropout, mlp_hidden, heads, symmetrize_edges)
    trainer = Trainer(model, lr=lr, weight_decay=wd, device=device)
    trainer.train(subgraph, epochs=epochs, early_stop=20)
    model.eval()
    with torch.no_grad():
        logits = model(subgraph.features.to(device), subgraph.edge_index.to(device))
        probs = F.softmax(logits, dim=1).cpu()
    return probs

def main():
    parser = argparse.ArgumentParser("Partitioned multi-GPU GNN training")
    parser.add_argument("--dataset", required=True, choices=["DBLP_bipartite", "ogbn-arxiv", "CiteSeer_bipartite"])
    parser.add_argument("--source", choices=["npz", "src"], default="npz")
    parser.add_argument("--model", choices=["gcn", "gat", "sage", "gin"], default="gcn")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=str, default="cuda:0,cuda:1,cuda:2,cuda:3", help="Comma-separated devices for parallel workers")
    parser.add_argument("--K", type=int, default=4, help="Number of partitions/workers")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--mlp", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--no_sym", action="store_true")
    parser.add_argument("--allow_large", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    dev_list = [d.strip() for d in args.devices.split(",") if d.strip()]
    if args.K != len(dev_list):
        print(f"Warning: K={args.K} inconsistent with device count {len(dev_list)}. Using first {min(args.K, len(dev_list))} devices.")
        args.K = min(args.K, len(dev_list))

    # Load graph
    if args.source == "npz":
        base_data_dir = Path(__file__).resolve().parent.parent / "data"
        if args.dataset == "DBLP_bipartite":
            path = base_data_dir / "DBLP/graph.npz"
        else:
            path = base_data_dir / "ogbn-arxiv/graph.npz"
        
        if path.exists():
            g = load_graph_npz(path)
        else:
             print(f"Warning: {path} not found, falling back to src")
             g = load_from_src(args.dataset, seed=args.seed)
    else:
        g = load_from_src(args.dataset, seed=args.seed)

    if args.dataset == "ogbn-products" and not args.allow_large:
        print("ogbn-products is very large. Add --allow_large to proceed.")
        return

    N = g.num_nodes
    in_dim = g.num_features
    num_classes = int(torch.unique(g.labels[g.labels >= 0]).numel())

    print(f"Partitioned Train on {args.dataset}: N={N}, F={in_dim}, E={g.num_edges}, C={num_classes}, Model={args.model}, K={args.K}")

    # Node partitions
    perm = torch.randperm(N)
    bins = perm % args.K
    subgraphs = []
    idx_list = []
    for i in range(args.K):
        nodes_i = perm[bins == i]
        mask_i = torch.zeros(N, dtype=torch.bool)
        mask_i[nodes_i] = True
        sub_i, idx_i = build_subgraph(g, mask_i)
        subgraphs.append(sub_i)
        idx_list.append(idx_i.cpu())

    # Parallel training
    futures = []
    probs_list = [None] * args.K
    with ProcessPoolExecutor(max_workers=args.K) as ex:
        for i in range(args.K):
            futures.append(ex.submit(
                _train_one_subgraph,
                subgraphs[i],
                args.model, in_dim, args.hidden, num_classes,
                args.dropout, args.mlp, args.heads, (not args.no_sym),
                args.lr, args.wd, args.epochs, dev_list[i],
            ))
        for i, f in enumerate(as_completed(futures)):
            # Collect in order of completion; map index by matching finished future
            idx = futures.index(f)
            probs_list[idx] = f.result()

    # Stitch global predictions
    C = num_classes
    probs_global = torch.zeros(N, C)
    for idx_i, probs_i in zip(idx_list, probs_list):
        probs_global[idx_i] = probs_i

    y = g.labels
    val_mask = g.val_mask
    test_mask = g.test_mask
    val_acc = accuracy(probs_global, y, val_mask)
    test_acc = accuracy(probs_global, y, test_mask)
    val_f1 = macro_f1(probs_global, y, val_mask)
    test_f1 = macro_f1(probs_global, y, test_mask)
    val_micro = micro_f1(probs_global, y, val_mask)
    test_micro = micro_f1(probs_global, y, test_mask)

    split = split_head_tail_by_degree(g.edge_index, g.num_nodes, head_ratio=0.2)
    val_group = compute_group_metrics(probs_global, y, val_mask, split)
    test_group = compute_group_metrics(probs_global, y, test_mask, split)

    print(f"ValAcc={val_acc:.4f}, TestAcc={test_acc:.4f}, "
          f"ValF1={val_f1:.4f}, TestF1={test_f1:.4f}, "
          f"ValMicro={val_micro:.4f}, TestMicro={test_micro:.4f}, "
          f"ValHeadAcc={val_group['HeadAcc']:.4f}, ValTailAcc={val_group['TailAcc']:.4f}, "
          f"TestHeadAcc={test_group['HeadAcc']:.4f}, TestTailAcc={test_group['TailAcc']:.4f}, "
          f"ValBalAcc={val_group['BalancedAcc']:.4f}, TestBalAcc={test_group['BalancedAcc']:.4f})")

if __name__ == "__main__":
    main()
