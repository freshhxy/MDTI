# utils/graph_utils.py
import dgl
import torch

def prep_subgraphs_with_self_loop(r_subgraphs, device=None):
    """确保每个子图都有自环；不改动 ndata['id']。"""
    if r_subgraphs is None:
        return None
    out = []
    for g in r_subgraphs:
        # 注意：保持有向，不做 to_bidirected
        if dgl.has_self_loop(g):
            g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        if device is not None:
            g = g.to(device)
        out.append(g)
    return out
