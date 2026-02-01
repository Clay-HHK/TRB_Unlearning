import argparse
from pathlib import Path
import numpy as np
import torch
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.seed import set_seed
from src.training.trainer import Trainer
from src.data.graph_data import GraphData
from src.data.loader import load_from_src
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

def main():
    parser = argparse.ArgumentParser("Train a minimal GNN on standardized graphs")
    parser.add_argument("--dataset", required=True, choices=["DBLP_bipartite", "ogbn-arxiv", "CiteSeer_bipartite", "ogbn-products"])
    parser.add_argument("--source", choices=["npz", "src"], default="npz", help="Load from graph.npz or src raw files")
    parser.add_argument("--model", choices=["gcn", "gat", "sage", "gin"], default="gcn")
    parser.add_argument("--heads", type=int, default=64, help="GAT heads")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--mlp", type=int, default=0, help="MLP head hidden dim (0 disables)")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=1e-3)
    parser.add_argument("--allow_large", action="store_true", help="Allow training on very large graphs (ogbn-products)")
    parser.add_argument("--no_sym", action="store_true", help="Disable edge symmetrization")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.source == "npz":
        base_data_dir = Path(__file__).resolve().parent.parent / "data"
        if args.dataset == "DBLP_bipartite":
            path = base_data_dir / "DBLP/graph.npz"
        elif args.dataset == "ogbn-arxiv":
            path = base_data_dir / "ogbn-arxiv/graph.npz"
        elif args.dataset == "ogbn-products":
             path = base_data_dir / "ogbn-products/graph.npz"
        else:
             path = base_data_dir / "CiteSeer/graph.npz"

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

    in_dim = g.num_features
    num_classes = int(torch.unique(g.labels[g.labels >= 0]).numel())

    if args.model == "gcn":
        from src.models.gcn import SimpleGCN
        model = SimpleGCN(in_dim=in_dim, hidden_dim=args.hidden, out_dim=num_classes, dropout=args.dropout, mlp_hidden=args.mlp, symmetrize_edges=(not args.no_sym))
    elif args.model == "gat":
        from src.models.gat import SimpleGAT
        model = SimpleGAT(
            in_dim=in_dim, hidden_dim=args.hidden, out_dim=num_classes,
            heads=args.heads, dropout=args.dropout, mlp_hidden=args.mlp,
            symmetrize_edges=(not args.no_sym),
        )
    elif args.model == "sage":
        from src.models.sage import GraphSAGE
        model = GraphSAGE(in_dim=in_dim, hidden_dim=args.hidden, out_dim=num_classes, dropout=args.dropout, mlp_hidden=args.mlp, symmetrize_edges=(not args.no_sym))
    else:
        from src.models.gin import GIN
        model = GIN(in_dim=in_dim, hidden_dim=args.hidden, out_dim=num_classes, dropout=args.dropout, mlp_hidden=args.mlp, symmetrize_edges=(not args.no_sym))

    print(f"Train on {args.dataset}: N={g.num_nodes}, F={g.num_features}, E={g.num_edges}, C={num_classes}, Model={args.model}")
    trainer = Trainer(model, lr=args.lr, weight_decay=args.wd, device=args.device)
    result = trainer.train(g, epochs=args.epochs, early_stop=20)

    print(f"ValAcc={result.best_val_acc:.4f}, TestAcc={result.best_test_acc:.4f}, "
          f"ValF1={result.best_val_f1:.4f}, TestF1={result.best_test_f1:.4f}, "
          f"ValHeadAcc={result.best_val_head_acc:.4f}, ValTailAcc={result.best_val_tail_acc:.4f}, "
          f"TestHeadAcc={result.best_test_head_acc:.4f}, TestTailAcc={result.best_test_tail_acc:.4f}, "
          f"ValHeadF1={result.best_val_head_f1:.4f}, ValTailF1={result.best_val_tail_f1:.4f}, "
          f"TestHeadF1={result.best_test_head_f1:.4f}, TestTailF1={result.best_test_tail_f1:.4f}, "
          f"ValBalAcc={result.best_val_balanced_acc:.4f}, TestBalAcc={result.best_test_balanced_acc:.4f}, "
          f"Epochs={result.epochs_run}")

if __name__ == "__main__":
    main()
