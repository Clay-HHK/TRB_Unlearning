import unittest
import torch
from src.models.gcn import SimpleGCN
from src.models.gat import SimpleGAT
from src.models.sage import GraphSAGE
from src.models.gin import GIN
from src.models.mlp import MLPClassifier

class TestModels(unittest.TestCase):
    def setUp(self):
        self.in_dim = 16
        self.hidden_dim = 32
        self.out_dim = 5
        self.num_nodes = 100
        self.num_edges = 500
        self.x = torch.randn(self.num_nodes, self.in_dim)
        self.edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))
        
    def test_gcn(self):
        model = SimpleGCN(self.in_dim, self.hidden_dim, self.out_dim)
        out = model(self.x, self.edge_index)
        self.assertEqual(out.shape, (self.num_nodes, self.out_dim))
        
    def test_gat(self):
        model = SimpleGAT(self.in_dim, self.hidden_dim, self.out_dim, heads=2)
        out = model(self.x, self.edge_index)
        self.assertEqual(out.shape, (self.num_nodes, self.out_dim))

    def test_sage(self):
        model = GraphSAGE(self.in_dim, self.hidden_dim, self.out_dim)
        out = model(self.x, self.edge_index)
        self.assertEqual(out.shape, (self.num_nodes, self.out_dim))

    def test_gin(self):
        model = GIN(self.in_dim, self.hidden_dim, self.out_dim)
        out = model(self.x, self.edge_index)
        self.assertEqual(out.shape, (self.num_nodes, self.out_dim))
        
    def test_mlp(self):
        model = MLPClassifier(self.in_dim, self.hidden_dim, self.out_dim)
        out = model(self.x)
        self.assertEqual(out.shape, (self.num_nodes, self.out_dim))

if __name__ == "__main__":
    unittest.main()
