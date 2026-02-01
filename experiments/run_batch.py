import argparse
import time
import subprocess
from pathlib import Path
import torch

def main():
    parser = argparse.ArgumentParser("Batch Unlearning Runner")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=10)
    args = parser.parse_args()

    # Determine range of nodes to delete
    nodes_to_delete = list(range(args.start_idx, args.start_idx + args.num_nodes))
    
    # Baselines
    baselines = ["retraining", "delete", "hctsa"] # "if", "bekm" omitted for brevity

    script_map = {
        "retraining": "run_baselines.py",
        "delete": "run_baselines.py",
        "hctsa": "run_hctsa.py"
    }

    base_dir = Path(__file__).resolve().parent

    for node in nodes_to_delete:
        print(f"=== Deleting Node {node} ===")
        for baseline in baselines:
            print(f"  >> Running {baseline}...")
            script = script_map[baseline]
            cmd = [
                "python", str(base_dir / script),
                "--dataset", args.dataset,
                "--device", args.device,
                "--delete_node", str(node),
                "--epochs", "100"
            ]
            if baseline in ["retraining", "delete"]:
                cmd.extend(["--baseline", baseline])
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running {baseline} for node {node}: {e}")

if __name__ == "__main__":
    main()
