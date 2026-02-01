import torch
from src.data.graph_data import GraphData

def remove_node(g: GraphData, u: int, drop_edges: bool = True) -> GraphData:
    """
    Removes a node from the graph.
    
    Args:
        g (GraphData): Input graph.
        u (int): Index of the node to remove.
        drop_edges (bool): Whether to remove edges connected to the node.
        
    Returns:
        GraphData: Modified graph.
    """
    ei = g.edge_index
    if drop_edges:
        keep = (ei[0] != u) & (ei[1] != u)
        new_ei = ei[:, keep]
    else:
        new_ei = ei
        
    new_labels = g.labels.clone()
    new_train = g.train_mask.clone()
    
    # Mark as invalid/removed
    new_labels[u] = -1
    new_train[u] = False
    
    return GraphData(
        features=g.features,
        labels=new_labels,
        edge_index=new_ei,
        train_mask=new_train,
        val_mask=g.val_mask,
        test_mask=g.test_mask,
    )

def remove_edge(g: GraphData, u: int, v: int) -> GraphData:
    """
    Removes an edge (u, v) from the graph (undirected removal).
    
    Args:
        g (GraphData): Input graph.
        u (int): Source node index.
        v (int): Target node index.
        
    Returns:
        GraphData: Modified graph.
    """
    ei = g.edge_index
    # Remove both (u, v) and (v, u)
    mask = ~(((ei[0] == u) & (ei[1] == v)) | ((ei[0] == v) & (ei[1] == u)))
    new_ei = ei[:, mask]
    
    return GraphData(
        features=g.features,
        labels=g.labels,
        edge_index=new_ei,
        train_mask=g.train_mask,
        val_mask=g.val_mask,
        test_mask=g.test_mask,
    )
