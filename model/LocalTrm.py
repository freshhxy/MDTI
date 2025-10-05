import torch
import torch.nn as nn


class LocalTransformer(nn.Module):
    def __init__(self, d_model, nhead, window_size=20, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.local_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _segment_sequence(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        segments = []
        for i in range(0, L - self.window_size + 1, self.window_size // 2):  # 50%重叠
            segments.append(x[:, i:i + self.window_size, :])
        return torch.stack(segments, dim=1)  # (B, num_segments, window_size, C)

    def forward(self, x):
        """
        输入: x -> (B, L, C)
        输出: (B, L, C)
        """
        residual = x
        segments = self._segment_sequence(x)  # (B, S, W, C)
        B, S, W, C = segments.shape

        # 处理每个局部窗口
        segments = segments.view(B * S, W, C)
        segments = self.norm(segments)
        attn_output, _ = self.local_attn(segments, segments, segments)
        segments = segments + self.dropout(attn_output)

        # 重组为完整序列（重叠部分取平均）
        output = torch.zeros_like(x)
        count = torch.zeros_like(x[:, :, 0])
        for i, seg in enumerate(segments.chunk(S, dim=0)):
            start = i * (self.window_size // 2)
            seg = seg.squeeze(0)
            output[:, start:start + self.window_size] += seg
            count[:, start:start + self.window_size] += 1

        output = output / count.unsqueeze(-1).clamp(min=1)
        return output + residual