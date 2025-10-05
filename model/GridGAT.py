# new_models/GridGAT.py
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GridGAT(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_heads: int, num_layers: int, dropout: float = 0.6):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        # 第一层 GAT
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout))
        # 中间层 GAT
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout))
        # 最后一层 GAT (通常只有一个头，以便输出维度是 out_channels)
        self.convs.append(GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index, batch):
        """
        Args:
            x (torch.Tensor): 节点特征，形状 (num_nodes, in_channels)。
            edge_index (torch.Tensor): 边索引，形状 (2, num_edges)。
            batch (torch.Tensor): 每个节点所属图的批次索引，形状 (num_nodes,)。
        Returns:
            torch.Tensor: 节点嵌入，形状 (num_nodes, out_channels)。
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:  # 最后一层通常不加激活函数
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x  # 如果你还用 batch，比如要 graph-level 表达，可以加 pooling
