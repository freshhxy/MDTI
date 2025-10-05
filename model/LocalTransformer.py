import math
import torch
import torch.nn as nn
class LocalTransformer(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8, window_size=50, step=25):
        super().__init__()
        self.window_size = window_size
        self.step = step

        # 局部注意力（滑动窗口内自注意力）
        self.local_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        """
        输入: x -> (B, C, L) 或 (B, N, C)
        输出: (B, C, L) 或 (B, N, C)
        """
        # 保存残差连接
        residual = x

        # 转换为序列格式 (B, L, C) 或 (B, N, C)
        if x.dim() == 3 and x.size(1) != self.window_size:
            x = x.permute(0, 2, 1)  # (B, C, L) → (B, L, C)

        # 滑动窗口处理
        B, L, C = x.shape
        processed_segments = []
        for i in range(0, L - self.window_size + 1, self.step):
            segment = x[:, i:i + self.window_size, :]

            # 局部自注意力
            segment = self.norm1(segment)
            attn_output, _ = self.local_attn(segment, segment, segment)
            segment = segment + attn_output

            # 前馈网络
            segment = self.norm2(segment)
            segment = segment + self.ffn(segment)

            processed_segments.append(segment)

        # 重叠部分取平均
        output = torch.zeros_like(x)
        count = torch.zeros_like(x)
        for i, seg in enumerate(processed_segments):
            start = i * self.step
            output[:, start:start + self.window_size] += seg
            count[:, start:start + self.window_size] += 1
        output = output / count.clamp(min=1)

        # 恢复原始维度
        if residual.dim() == 3 and residual.size(1) != self.window_size:
            output = output.permute(0, 2, 1)  # (B, L, C) → (B, C, L)

        return output + residual  # 残差连接