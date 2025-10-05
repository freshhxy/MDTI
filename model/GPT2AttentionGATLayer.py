import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config

class GPT2AttentionGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size=128, n_layer=2, n_head=4, max_neighbors=20):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.max_neighbors = max_neighbors

        # 初始化一个未预训练的 GPT-2 模型
        config = GPT2Config(
            n_embd=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            n_positions=max_neighbors + 1,  # +1 for self node
            n_ctx=max_neighbors + 1,
            vocab_size=1  # 不用embedding，所以随便设一个
        )
        self.gpt2 = GPT2Model(config)

        # 输入投影: 将节点特征投影到 GPT2 的 hidden size
        self.input_proj = nn.Linear(in_dim, hidden_size)

        # 输出投影: 从 GPT2 的输出映射到期望输出维度
        self.output_proj = nn.Linear(hidden_size, out_dim)

    # def forward(self, x, adj):
    #     """
    #     x: [N, in_dim] - 所有节点的输入特征
    #     adj: List[List[int]] - 每个节点的邻接节点索引
    #     """
    #     device = x.device
    #     N = x.size(0)
    #     out = torch.zeros(N, self.out_dim, device=device)
    #
    #     for i in range(N):
    #         neighbors = adj[i][:self.max_neighbors]  # 限制最大邻居数
    #         input_nodes = [i] + neighbors  # 将自身加入序列开头
    #
    #         # 构造输入特征序列 [len, in_dim] → [1, len, hidden_size]
    #         input_features = x[input_nodes]
    #         input_embeds = self.input_proj(input_features).unsqueeze(0)
    #
    #         # 用 GPT2 做 attention
    #         output = self.gpt2(inputs_embeds=input_embeds)
    #         central_node_output = output.last_hidden_state[0, 0]  # 取第一个位置（中心节点）
    #
    #         out[i] = self.output_proj(central_node_output)
    #
    #     return out
    def forward(self, x, adj):
        """
        x: [N, in_dim]
        adj: List[List[int]]
        """
        device = x.device
        N = x.size(0)
        input_node_lists = []

        for i in range(N):
            neighbors = adj[i][:self.max_neighbors]
            input_nodes = [i] + neighbors
            while len(input_nodes) < self.max_neighbors + 1:
                input_nodes.append(i)  # padding with self
            input_node_lists.append(input_nodes)

        # shape: [N, max_neighbors+1, in_dim]
        subgraph_feats = torch.stack([
            x[nodes] for nodes in input_node_lists
        ], dim=0)

        # 投影 → GPT2 输入
        input_embeds = self.input_proj(subgraph_feats)  # [N, L, hidden_size]

        self.gpt2.eval()
        with torch.no_grad():  # 可选，如果是 inference
            gpt_output = self.gpt2(inputs_embeds=input_embeds)

        # 取中心节点输出
        central_output = gpt_output.last_hidden_state[:, 0]  # [N, hidden_size]

        out = self.output_proj(central_output)  # [N, out_dim]
        return out
