# RB-Unlearning: Robust Graph Unlearning via Hierarchical Clustering

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-1.12%2B-orange)

This repository contains the official implementation of **RB-Unlearning**, a robust and efficient graph unlearning framework designed for Graph Neural Networks (GNNs). It supports various unlearning scenarios (node deletion, edge deletion) and provides comprehensive evaluation metrics including model utility, unlearning efficiency, and defense against Membership Inference Attacks (MIA).

## 🌟 Features

- **Modular Design**: Clear separation of models, data loaders, training logic, and unlearning algorithms.
- **Supported Models**: GCN, GAT, GraphSAGE, GIN, MLP.
- **Unlearning Algorithms**:
  - **Retraining** (Exact Unlearning baseline)
  - **Gradient Ascent** (Delete Similarity)
  - **Influence Functions**
  - **BEKM** (K-Means based partitioning)
  - **HCTSA** (Ours - Hierarchical Clustering & Structure Aware)
- **Evaluation**: Accuracy, F1-Score, Training/Unlearning Time, Membership Inference Attack (MIA) AUC.
- **Reproducibility**: Standardized seeds and configuration management.

## 📂 Directory Structure

```
rb_unlearning/
├── data/               # Dataset storage and preprocessing scripts
├── experiments/        # Experiment scripts (train, unlearn, baselines)
├── models/             # Model definitions (GCN, GAT, SAGE, GIN)
├── results/            # Experiment logs and CSV results
├── src/                # Core source code
│   ├── attack/         # Membership Inference Attacks (MIA)
│   ├── datasets/       # Data loaders and graph structures
│   ├── evaluation/     # Metrics (Accuracy, F1, Long-tail)
│   ├── fusion/         # HCTSA fusion logic
│   ├── training/       # Training loops and utils
│   ├── unlearning/     # Unlearning algorithms implementation
│   └── utils/          # Helper functions (seed, graph ops)
├── tests/              # Unit and integration tests
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## 🛠️ Installation

1. **Clone the repository** (if applicable) or navigate to the directory.
2. **Create a virtual environment** (recommended):
   ```bash
   conda create -n rb_unlearning python=3.9
   conda activate rb_unlearning
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure PyTorch and PyTorch Geometric are installed compatible with your CUDA version.*

## 🚀 Usage

### 1. Data Preparation
The code supports `DBLP_bipartite`, `ogbn-arxiv`, `ogbn-products`, and `CiteSeer_bipartite`.
To pre-process a dataset into a standard `.npz` format:
```bash
python data/preprocess_data.py --dataset DBLP_bipartite
```

### 2. Training a Baseline Model
Train a standard GNN model on the full graph:
```bash
python experiments/train_baseline.py --dataset DBLP_bipartite --model gcn --epochs 200
```

### 3. Running Unlearning Experiments
Run a specific unlearning baseline (e.g., Retraining, Delete Similarity, Influence Function):
```bash
python experiments/run_baselines.py \
  --dataset DBLP_bipartite \
  --baseline retraining \
  --delete_node 10 \
  --model gcn \
  --epochs 100
```

### 4. Running HCTSA (Ours)
Run the proposed HCTSA method:
```bash
python experiments/run_hctsa.py \
  --dataset DBLP_bipartite \
  --partition hctsa \
  --delete_node 10 \
  --model gcn \
  --K 4
```

### 5. Batch Experiments
Run unlearning on a range of nodes sequentially:
```bash
python experiments/run_batch.py --dataset DBLP_bipartite --start_idx 0 --num_nodes 5
```

## 📊 Reproduction

To reproduce the main results of the paper, use the following commands:

**Table 1: Utility & Efficiency Comparison**
```bash
# HCTSA
python experiments/run_hctsa.py --dataset ogbn-arxiv --partition hctsa --K 8 --delete_node 0

# Retraining
python experiments/run_baselines.py --dataset ogbn-arxiv --baseline retraining --delete_node 0

# Influence Function
python experiments/run_baselines.py --dataset ogbn-arxiv --baseline if --delete_node 0
```

Results will be saved in `results/`.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Citation

If you find this code useful, please cite our paper:
(Placeholder for citation)
