import torch
from transformers import GPT2Model, GPT2Config
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_batch


class GridGATGPT2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_heads, num_layers, gpt2_hidden_size=768, gpt2_layers=4, dropout=0.6):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout))
        self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False, dropout=dropout))

        self.linear_proj = nn.Linear(hidden_channels, gpt2_hidden_size)
        config = GPT2Config(
            n_embd=gpt2_hidden_size,
            n_layer=gpt2_layers,
            n_head=12,
            n_positions=512  # 保持与实际截断长度一致，或根据需要调整
        )
        self.gpt2 = GPT2Model(config)
        self.gpt2.gradient_checkpointing_enable()  # 开启 gradient checkpointing
        self.linear_out = nn.Linear(gpt2_hidden_size, out_channels)

    def forward(self, x, edge_index, batch=None):
        original_num_nodes = x.size(0)  # 记录原始总节点数

        # 1. GAT 层处理
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # 2. 投影到 GPT-2 输入维度
        x_proj = self.linear_proj(x)  # (original_num_nodes, gpt2_hidden_size)

        # 3. 转换为 GPT-2 期望的密集批次表示 (batch_size, max_nodes_in_batch, feature_dim)
        # mask_dense: (batch_size, max_nodes_in_batch)
        if batch is not None:
            x_proj_dense, mask_dense = to_dense_batch(x_proj, batch)
        else:
            # 单图情况
            x_proj_dense = x_proj.unsqueeze(0)
            mask_dense = torch.ones(1, original_num_nodes, dtype=torch.bool, device=x.device)

        # 确保连续内存
        x_proj_dense = x_proj_dense.contiguous()

        # 4. 根据 GPT-2 的最大序列长度进行截断
        # max_gpt2_seq_len 来自于 __init__ 中 GPT2Config 的 n_positions
        max_gpt2_seq_len = self.gpt2.config.n_positions  # 使用 GPT2 配置中的最大长度

        truncated_seq_len = min(x_proj_dense.size(1), max_gpt2_seq_len)

        # 对输入和对应的 mask 进行截断
        x_proj_batch_truncated = x_proj_dense[:, :truncated_seq_len, :]
        mask_batch_truncated = mask_dense[:, :truncated_seq_len]

        if x_proj_dense.size(1) > max_gpt2_seq_len:
            print(
                f"[WARNING] GPT2 input sequence too long: {x_proj_dense.size(1)}. Truncating to {truncated_seq_len} tokens."
            )

        # 5. GPT-2 前向传播
        # attention_mask 会确保 GPT-2 只关注实际的节点，忽略填充部分
        gpt2_out = self.gpt2(inputs_embeds=x_proj_batch_truncated,
                             attention_mask=mask_batch_truncated).last_hidden_state
        # gpt2_out 形状: (batch_size, truncated_seq_len, gpt2_hidden_size)

        # 6. 从 GPT-2 输出中提取有效节点嵌入（展平）
        # `processed_flat_embeddings` 包含所有批次中，经过 GPT-2 处理的真实节点嵌入
        processed_flat_embeddings = gpt2_out[mask_batch_truncated]
        # `processed_flat_embeddings` 形状: (所有批次中被处理的节点总数, gpt2_hidden_size)

        # 7. 映射到最终输出维度
        out_flat = self.linear_out(processed_flat_embeddings)
        # `out_flat` 形状: (所有批次中被处理的节点总数, out_channels)

        # 8. 将输出填充回原始节点数量
        # 创建一个与原始节点数相同，用零填充的张量
        final_output_embeddings = torch.zeros(
            original_num_nodes,
            self.linear_out.out_features,  # 从 linear_out 层获取正确的输出维度
            device=x.device,
            dtype=out_flat.dtype
        )

        # 精确地找到哪些原始节点被处理了，并回填
        # 这一部分是处理 `IndexError` 的关键

        # 如果没有批处理，直接填充前 N 个节点
        if batch is None:
            final_output_embeddings[:out_flat.size(0)] = out_flat
        else:
            # 1. 确保原始节点批次信息
            # `torch.bincount(batch)` 得到每个图的节点数量
            # `cumsum` 得到每个图在原始 `x` 中的结束索引
            node_counts_per_graph = torch.bincount(batch)
            # `graph_node_starts` 存储每个图在原始 `x` 中的起始索引
            graph_node_starts = torch.cat([torch.tensor([0], device=x.device), node_counts_per_graph.cumsum(0)[:-1]])

            # `current_out_flat_idx` 跟踪 `out_flat` 中当前已使用的位置
            current_out_flat_idx = 0

            # 遍历每个批次（图）
            for i_batch in range(mask_batch_truncated.size(0)):
                # 得到当前图在截断后的密集批次中的有效节点掩码
                current_graph_mask = mask_batch_truncated[i_batch]  # shape: (truncated_seq_len,)

                # 找出 `current_graph_mask` 中为 True 的局部索引（在截断后的密集行中）
                local_dense_indices = torch.nonzero(current_graph_mask).squeeze(1)

                # 计算当前图在原始 `x` 中的真实节点数量
                num_original_nodes_in_current_graph = node_counts_per_graph[i_batch].item()

                # 筛选出 `local_dense_indices` 中那些真正属于当前图原始节点的索引
                # 避免包含 `to_dense_batch` 填充的索引，或超出原始图节点范围的索引
                valid_local_dense_indices = local_dense_indices[
                    local_dense_indices < num_original_nodes_in_current_graph]

                if valid_local_dense_indices.numel() > 0:  # 如果当前图有被处理的有效节点
                    # 计算这些节点在原始 `x` 中的全局索引
                    # `graph_node_starts[i_batch]` 是当前图在原始 `x` 中的起始全局索引
                    global_indices_for_this_graph = graph_node_starts[i_batch] + valid_local_dense_indices

                    # 获取 `out_flat` 中对应当前图的嵌入
                    num_processed_in_this_segment = valid_local_dense_indices.numel()
                    embeddings_to_fill = out_flat[
                                         current_out_flat_idx: current_out_flat_idx + num_processed_in_this_segment]

                    # 将嵌入填充到 `final_output_embeddings` 的正确位置
                    final_output_embeddings[global_indices_for_this_graph] = embeddings_to_fill

                    # 更新 `out_flat` 的索引，以便下一个图使用
                    current_out_flat_idx += num_processed_in_this_segment

        return final_output_embeddings
