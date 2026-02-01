import unittest
import torch
from src.data.graph_data import GraphData
from src.unlearning.utils import remove_node, remove_edge

class TestUnlearningUtils(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 10
        self.num_edges = 20
        self.features = torch.randn(self.num_nodes, 5)
        self.labels = torch.randint(0, 3, (self.num_nodes,))
        self.edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))
        self.train_mask = torch.ones(self.num_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        
        self.g = GraphData(
            features=self.features,
            labels=self.labels,
            edge_index=self.edge_index,
            train_mask=self.train_mask,
            val_mask=self.val_mask,
            test_mask=self.test_mask
        )

    def test_remove_node(self):
        u = 0
        g_new = remove_node(self.g, u, drop_edges=True)
        self.assertEqual(g_new.labels[u].item(), -1)
        self.assertFalse(g_new.train_mask[u].item())
        
        # Check edges connected to u are removed
        has_u = ((g_new.edge_index[0] == u) | (g_new.edge_index[1] == u)).any().item()
        self.assertFalse(has_u)

    def test_remove_edge(self):
        # Find an existing edge
        u, v = int(self.edge_index[0, 0]), int(self.edge_index[1, 0])
        g_new = remove_edge(self.g, u, v)
        
        # Check edge (u, v) is gone
        has_uv = ((g_new.edge_index[0] == u) & (g_new.edge_index[1] == v)).any().item()
        has_vu = ((g_new.edge_index[0] == v) & (g_new.edge_index[1] == u)).any().item()
        self.assertFalse(has_uv)
        self.assertFalse(has_vu)

if __name__ == "__main__":
    unittest.main()
