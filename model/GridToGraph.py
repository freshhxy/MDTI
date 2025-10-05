# new_models/GridToGraph.py
import torch
import torch.nn as nn
from torch_geometric.data import Data

class GridToGraph(nn.Module):
    def __init__(self, in_channels: int, grid_h: int, grid_w: int):
        super().__init__()
        self.in_channels = in_channels
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_nodes = grid_h * grid_w

        # Optionally, if you want to initially embed pixel features before passing to GAT
        # self.pixel_embed = nn.Linear(in_channels, some_embedding_dim)

    def forward(self, grid_image: torch.Tensor):
        """
        将批量的 grid_image 转换为 PyTorch Geometric 的 Batch 对象。
        Args:
            grid_image (torch.Tensor): 批量的 grid images. Shape: (B, C, H, W)
        Returns:
            torch_geometric.data.Batch: 包含节点特征 (x) 和边索引 (edge_index) 的图数据批次。
        """
        # 注意: grid_image 的输入预期是 (B, C, H, W)
        B, C, H, W = grid_image.shape
        if H != self.grid_h or W != self.grid_w or C != self.in_channels:
            raise ValueError(f"Input grid_image shape {grid_image.shape} does not match "
                             f"initialized dimensions (B, {self.in_channels}, {self.grid_h}, {self.grid_w})")

        data_list = []
        for i in range(B):
            single_grid = grid_image[i] # (C, H, W)

            # 节点特征: 将 (C, H, W) 展平为 (num_nodes, C)
            x = single_grid.permute(1, 2, 0).reshape(self.num_nodes, C)
            # if hasattr(self, 'pixel_embed'):
            #     x = self.pixel_embed(x)

            edge_indices = []
            for r in range(H):
                for c in range(W):
                    node_idx = r * W + c # 计算当前像素的节点索引
                    # 定义 8-连通性邻居
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            # if dr == 0 and dc == 0: # 跳过自身
                            #     continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W: # 检查是否在网格内
                                neighbor_idx = nr * W + nc
                                edge_indices.append((node_idx, neighbor_idx))

            # 边索引: 转换为 (2, num_edges) 格式
            edge_index = torch.tensor(edge_indices, dtype=torch.long, device=grid_image.device).t().contiguous()
            data_list.append(Data(x=x, edge_index=edge_index))

        # PyTorch Geometric 的 Batch 对象能够处理多个图
        from torch_geometric.data import Batch
        return Batch.from_data_list(data_list)
