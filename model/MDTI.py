import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from config.config import Config
from model.GridTrm import GridTrm
from model.RoadTrm import RoadTrm
from model.RoadGNN import RoadGNN
from model.GridConv import GridConv
from model.InterTrm import InterTrm

# 导入新定义的 GAT 相关类
from model.GridToGraph import GridToGraph

import h5py
from torch_geometric.nn import GATConv


def create_unified_mask(road_traj, mask_ratio=0.15):
    """
    创建BERT式的统一随机mask - 所有batch使用相同的随机位置

    Args:
        road_traj: [batch_size, seq_len] 原始road轨迹
        mask_ratio: float, mask的比例，默认15%

    Returns:
        mask_positions: [num_mask] 被mask的位置索引
        original_tokens: [batch_size * num_mask] 被mask位置的原始token
    """
    batch_size, seq_len = road_traj.shape
    device = road_traj.device

    # 生成随机mask位置（所有batch共享）
    num_mask = max(1, int(seq_len * mask_ratio))
    mask_positions = torch.randperm(seq_len, device=device)[:num_mask]
    mask_positions = mask_positions.sort()[0]  # 排序便于调试

    # 提取被mask位置的原始tokens
    original_tokens = road_traj[:, mask_positions]  # [batch_size, num_mask]
    original_tokens = original_tokens.reshape(-1)  # [batch_size * num_mask]

    return mask_positions, original_tokens

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 300):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000)) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1)])

class GraphEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(GraphEncoder, self).__init__()
        # update road edge features using GAT
        self.layer1 = GATConv(input_size, output_size)
        self.layer2 = GATConv(output_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.activation(self.layer1(x, edge_index))
        x = self.activation(self.layer2(x, edge_index))
        return x

class MDTI(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe = PositionalEncoding(Config.hidden_emb_dim, Config.pe_dropout)
        # self.pattern_keys = pattern_keys
        # Grid encoder
        self.grid_cls_token = nn.Parameter(torch.randn(Config.hidden_emb_dim))
        self.grid_padding_token = nn.Parameter(torch.zeros(Config.hidden_emb_dim), requires_grad=False)

        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # --- 将 GridConv 替换为 GridToGraph 和 GridGAT ---
        # 假设 Config 中定义了 grid_image 的高和宽
        self.grid_to_graph = GridToGraph(Config.grid_in_channel, Config.grid_H, Config.grid_W)
        # self.grid_gat = GridGAT(
        #     in_channels=Config.in_channel,  # 节点特征的维度 (即像素的通道数)
        #     hidden_channels=Config.gat_hidden_channels,  # GAT 隐藏层维度
        #     out_channels=Config.grid_out_channel,  # GAT 输出维度，应与原 CNN 输出匹配
        #     num_heads=Config.gat_num_heads,  # GAT 注意力头数量
        #     num_layers=Config.gat_num_layers,  # GAT 层数
        #     dropout=Config.gat_dropout  # GAT dropout
        # )
        self.grid_gat = GraphEncoder(64, 128)
        # # --- 结束 GAT 替换 ---
        if Config.grid_out_channel != Config.hidden_emb_dim:
            self.grid_gat_proj_to_transformer = nn.Linear(128, 256)
        else:
            self.grid_gat_proj_to_transformer = nn.Identity()  # 如果维度相同，则使用恒等映射

        self.grid_conv = GridConv(Config.grid_in_channel, Config.grid_out_channel)
        # --- Prompt 嵌入融合的投影层 ---
        # 这里假设 prompt_emb 最终需要融合到 hidden_emb_dim 的维度
        # 如果 Config.hidden_emb_dim 不等于 768（GPT-2的默认输出维度）
        if Config.hidden_emb_dim != 768:  # 假设 GPT-2 输出维度是 768
            self.prompt_fusion_proj = nn.Linear(768, Config.hidden_emb_dim)
        else:
            self.prompt_fusion_proj = nn.Identity()  # 维度匹配，使用恒等映射

        self.grid_enc = GridTrm(
            Config.hidden_emb_dim,
            Config.grid_ffn_dim,
            Config.grid_trm_head,
            Config.grid_trm_layer,
            Config.grid_trm_dropout,
        )
        self.fusion_linear = nn.Linear(Config.hidden_emb_dim + 4, Config.hidden_emb_dim)
        self.grid_raw_proj = nn.Linear(Config.grid_in_channel, Config.grid_out_channel)
        # road encoder
        self.week_emb_layer = nn.Embedding(7 + 1, Config.hidden_emb_dim, padding_idx=0)
        self.minute_emb_layer = nn.Embedding(1440 + 1, Config.hidden_emb_dim, padding_idx=0)

        self.road_cls_token = nn.Parameter(torch.randn(Config.hidden_emb_dim))
        self.road_padding_token = nn.Parameter(torch.zeros(Config.hidden_emb_dim), requires_grad=False)
        self.road_mask_token = nn.Parameter(torch.randn(Config.hidden_emb_dim))

        self.road_emb_layer = RoadGNN(
            Config.g_fea_size,
            Config.g_dim_per_layer,
            Config.g_heads_per_layer,
            Config.g_num_layers,
            Config.g_dropout
        )
        self.type_emb_layer = nn.Embedding(Config.road_type + 1, Config.hidden_emb_dim, padding_idx=0)
        self.road_enc = RoadTrm(
            Config.hidden_emb_dim,
            Config.road_ffn_dim,
            Config.road_trm_head,
            Config.road_trm_layer,
            Config.road_trm_dropout,
        )

        # InterLayer
        self.inter_layer = InterTrm(
            Config.out_emb_dim,
            Config.inter_ffn_dim,
            Config.inter_trm_head,
            Config.inter_trm_layer,
            Config.inter_trm_dropout,
        )

        # cl_linear
        self.grid_cl_linear = nn.Linear(
            Config.hidden_emb_dim,
            Config.out_emb_dim
        )

        self.road_cl_linear = nn.Linear(
            Config.hidden_emb_dim,
            Config.out_emb_dim
        )

        self.temp = nn.Parameter(torch.ones([]) * 0.07)

        # mlm_linear
        self.road_mlm_linear = nn.Linear(
            Config.out_emb_dim,
            Config.road_num + len(Config.road_special_tokens)
        )

        self.prompt_cache = {}


    def load_prompt_embedding_from_h5(self, file_path, device, indices=None):
        with h5py.File(file_path, 'r') as f:
            if 'embeddings' not in f:
                raise ValueError(f"'embeddings' not found in {file_path}")

            if indices is not None:
                indices = indices.cpu().numpy()
                data = f['embeddings'][indices]
            else:
                raise ValueError("load_prompt_embedding_from_h5: you must pass indices for current batch!")
        return torch.tensor(data, device=device, dtype=torch.float32)

    def forward(self, grid_data, road_data):
        # # ==== Road input ====
        # g_input_feature, g_edge_index = road_data['g_input_feature'], road_data['g_edge_index']
        # road_traj, mask_road_index = road_data['road_traj'], road_data['mask_road_index']
        # road_weeks, road_minutes = road_data['road_weeks'], road_data['road_minutes']
        # road_type = road_data['road_type']
        # ==== Road input ====
        g_input_feature, g_edge_index = road_data['g_input_feature'], road_data['g_edge_index']
        road_traj = road_data['road_traj']
        road_weeks, road_minutes = road_data['road_weeks'], road_data['road_minutes']
        road_type = road_data['road_type']
        # ==== Grid input ====
        grid_image = grid_data['grid_image']
        grid_traj = grid_data['grid_traj']
        grid_time_emb = grid_data['grid_time_emb']
        grid_feature = grid_data['grid_feature']
        # # ==== Grid / Road masks ====
        # grid_padding_mask = (grid_traj > 0)  # [B,L]
        # road_padding_mask = (road_traj > 0)  # [B,L]
        # ==== Road encoder ====
        road_weeks_emb = self.week_emb_layer(road_weeks)
        road_minutes_emb = self.minute_emb_layer(road_minutes)

        g_emb = self.road_emb_layer(g_input_feature, g_edge_index)
        g_emb = torch.vstack([self.road_padding_token, self.road_cls_token, self.road_mask_token, g_emb])
        road_seq_emb = g_emb[road_traj] + road_weeks_emb + road_minutes_emb
        road_seq_emb = self.pe(road_seq_emb)
        road_type_emb = self.pe(self.type_emb_layer(road_type))
        road_padding_mask = road_traj > 0

        road_seq_emb = self.road_enc(
            src=road_seq_emb,
            type_src=road_type_emb,
            mask=road_padding_mask
        )
        road_seq_emb = self.road_cl_linear(road_seq_emb)
        road_traj_emb = road_seq_emb[:, 0]  # [CLS]

        # ==== Grid encoder (CNN → GAT) ====
        if grid_image.dim() == 3:
            grid_image = grid_image.unsqueeze(0)  # (1, H, W, C)
        grid_image = grid_image.permute(0, 3, 1, 2)  # (B, C, H, W)
        grid_image = self.transform(grid_image)

        # cnn_feat = self.grid_conv(grid_image)  # (B, D, H, W)
        # B, D, H, W = cnn_feat.shape
        # cnn_feat_flat = cnn_feat.permute(0, 2, 3, 1).reshape(B * H * W, D)
        #####
        # 使用 grid_image 原始值生成图结构
        grid_graph_batch = self.grid_to_graph(grid_image)  # 使用 [B, C, H, W] 格式的图像生成图

        # 准备 GAT 输入
        B, C, H, W = grid_image.shape
        grid_image_flat = grid_image.permute(0, 2, 3, 1).reshape(B * H * W, C)  # [B*H*W, C]

        # 使用投影层对原始像素值投影（如果维度不同）
        grid_feat_flat = self.grid_raw_proj(grid_image_flat)  # [B*H*W, grid_out_channel]

        # GAT 编码
        edge_index = grid_graph_batch.edge_index
        grid_node_emb = self.grid_gat(grid_feat_flat, edge_index)  # 输出 [B*H*W, D']
        grid_node_emb = self.grid_gat_proj_to_transformer(grid_node_emb)
        ###
        # edge_index = grid_graph_batch.edge_index
        # batch = grid_graph_batch.batch
        # grid_node_emb = self.grid_gat(cnn_feat_flat, edge_index)  # (B*H*W, D')
        # grid_node_emb = self.grid_gat_proj_to_transformer(grid_node_emb)
        ###单gat
        # B, C, H, W = grid_image.shape
        # grid_image = grid_image.permute(0, 2, 3, 1)  # (B, H, W, C)
        # grid_feat_flat = grid_image.reshape(B * H * W, C)
        # grid_feat_flat = self.grid_raw_proj(grid_feat_flat)  # (B*H*W, gat_input_dim)
        # grid_graph_batch = self.grid_to_graph(grid_image.permute(0, 3, 1, 2))  # 若 `GridToGraph` 接受的是 [B, C, H, W]
        # edge_index = grid_graph_batch.edge_index
        # batch = grid_graph_batch.batch
        # grid_node_emb = self.grid_gat(grid_feat_flat, edge_index, batch)
        ###
        full_grid_emb = torch.cat([
            self.grid_padding_token.unsqueeze(0),
            self.grid_cls_token.unsqueeze(0),
            grid_node_emb
        ], dim=0)

        grid_seq_emb = full_grid_emb[grid_traj]  # (B, L, D)
        grid_seq_emb = torch.cat([grid_seq_emb, grid_feature], dim=-1)
        grid_seq_emb = self.fusion_linear(grid_seq_emb) + grid_time_emb
        grid_seq_emb = self.pe(grid_seq_emb)

        # ==== Prompt embedding 融合 ====
        B = grid_data['grid_traj'].shape[0]
        batch_index = torch.arange(B, device=grid_traj.device)
        file_path = f"./model/Embeddings/porto/train.h5"
        prompt_emb = self.load_prompt_embedding_from_h5(file_path, device=Config.device, indices=batch_index)
        projected_prompt_emb = self.prompt_fusion_proj(prompt_emb)  # (B, L-1, D)

        prompt_len = projected_prompt_emb.shape[1]
        target_len = grid_seq_emb.shape[1] - 1
        if prompt_len == target_len:
            grid_seq_emb[:, 1:, :] += projected_prompt_emb
        elif prompt_len > target_len:
            grid_seq_emb[:, 1:, :] += projected_prompt_emb[:, :target_len, :]
        else:
            pad_len = target_len - prompt_len
            pad_tensor = torch.zeros(B, pad_len, projected_prompt_emb.size(2), device=projected_prompt_emb.device)
            grid_seq_emb[:, 1:, :] += torch.cat([projected_prompt_emb, pad_tensor], dim=1)

        # ==== Transformer + contrastive loss ====
        grid_padding_mask = grid_traj > 0
        grid_seq_emb = self.grid_enc(grid_seq_emb, grid_padding_mask)
        grid_seq_emb = self.grid_cl_linear(grid_seq_emb)
        grid_traj_emb = grid_seq_emb[:, 0]

        road_e = F.normalize(road_traj_emb, dim=-1)
        grid_e = F.normalize(grid_traj_emb, dim=-1)
        logits = torch.matmul(grid_e, road_e.T) / self.temp

        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        cl_loss = (loss_i + loss_t) / 2

        # ==== MLM loss ====
        # === 新的mask逻辑 ===
        mask_positions, original_tokens = create_unified_mask(road_traj, mask_ratio=0.15)

        # 创建masked轨迹
        mask_road_traj = road_traj.clone()
        mask_road_traj[:, mask_positions] = Config.road_special_tokens['mask_token']

        # 后续处理保持不变
        mask_road_seq_emb = g_emb[mask_road_traj] + road_weeks_emb + road_minutes_emb
        mask_road_seq_emb = self.pe(mask_road_seq_emb)

        mask_road_seq_emb = self.road_enc(
            src=mask_road_seq_emb,
            type_src=road_type_emb,
            mask=road_padding_mask,
        )
        mask_road_seq_emb = self.road_cl_linear(mask_road_seq_emb)

        fusion_seq_emb = self.inter_layer(
            src=mask_road_seq_emb,
            src_mask=road_padding_mask,
            memory=grid_seq_emb,
            memory_mask=grid_padding_mask
        )

        # === MLM预测和损失计算 ===
        mlm_prediction = fusion_seq_emb[:, mask_positions]  # [batch_size, num_mask, embed_dim]
        mlm_prediction = mlm_prediction.reshape(-1, mlm_prediction.size(-1))  # [batch_size*num_mask, embed_dim]
        mlm_prediction = self.road_mlm_linear(mlm_prediction)

        # MLM loss
        mlm_loss = F.cross_entropy(mlm_prediction, original_tokens)

        return cl_loss, mlm_loss, mlm_prediction

    # # ==== Grid encoder (CNN → GAT) ====
        # if grid_image.dim() == 3:
        #     grid_image = grid_image.unsqueeze(0)  # (1, H, W, C)
        # grid_image = grid_image.permute(0, 3, 1, 2)  # (B, C, H, W)
        # grid_image = self.transform(grid_image)
        #
        # # cnn_feat = self.grid_conv(grid_image)  # (B, D, H, W)
        # # B, D, H, W = cnn_feat.shape
        # # cnn_feat_flat = cnn_feat.permute(0, 2, 3, 1).reshape(B * H * W, D)
        # #####
        # # 使用 grid_image 原始值生成图结构
        # grid_graph_batch = self.grid_to_graph(grid_image)  # 使用 [B, C, H, W] 格式的图像生成图
        #
        # # 准备 GAT 输入
        # B, C, H, W = grid_image.shape
        # grid_image_flat = grid_image.permute(0, 2, 3, 1).reshape(B * H * W, C)  # [B*H*W, C]
        #
        # # 使用投影层对原始像素值投影（如果维度不同）
        # grid_feat_flat = self.grid_raw_proj(grid_image_flat)  # [B*H*W, grid_out_channel]
        #
        # # GAT 编码
        # edge_index = grid_graph_batch.edge_index
        # grid_node_emb = self.grid_gat(grid_feat_flat, edge_index)  # 输出 [B*H*W, D']
        # grid_node_emb = self.grid_gat_proj_to_transformer(grid_node_emb)
        # ###
        # # edge_index = grid_graph_batch.edge_index
        # # batch = grid_graph_batch.batch
        # # grid_node_emb = self.grid_gat(cnn_feat_flat, edge_index)  # (B*H*W, D')
        # # grid_node_emb = self.grid_gat_proj_to_transformer(grid_node_emb)
        # ###单gat
        # # B, C, H, W = grid_image.shape
        # # grid_image = grid_image.permute(0, 2, 3, 1)  # (B, H, W, C)
        # # grid_feat_flat = grid_image.reshape(B * H * W, C)
        # # grid_feat_flat = self.grid_raw_proj(grid_feat_flat)  # (B*H*W, gat_input_dim)
        # # grid_graph_batch = self.grid_to_graph(grid_image.permute(0, 3, 1, 2))  # 若 `GridToGraph` 接受的是 [B, C, H, W]
        # # edge_index = grid_graph_batch.edge_index
        # # batch = grid_graph_batch.batch
        # # grid_node_emb = self.grid_gat(grid_feat_flat, edge_index, batch)
        # ###
        # full_grid_emb = torch.cat([
        #     self.grid_padding_token.unsqueeze(0),
        #     self.grid_cls_token.unsqueeze(0),
        #     grid_node_emb
        # ], dim=0)
        #
        # grid_seq_emb = full_grid_emb[grid_traj]  # (B, L, D)
        # grid_seq_emb = torch.cat([grid_seq_emb, grid_feature], dim=-1)
        # grid_seq_emb = self.fusion_linear(grid_seq_emb) + grid_time_emb
        # # grid_seq_emb = self.pe(grid_seq_emb)
        # grid_seq_emb = self.pe(grid_seq_emb, padding_mask=grid_padding_mask)
        # # ==== Prompt embedding 融合 ====
        # B = grid_data['grid_traj'].shape[0]
        # batch_index = torch.arange(B, device=grid_traj.device)

        # prompt_emb = self.load_prompt_embedding_from_h5(file_path, device=Config.device, indices=batch_index)
        # projected_prompt_emb = self.prompt_fusion_proj(prompt_emb)  # (B, L-1, D)
        #
        # prompt_len = projected_prompt_emb.shape[1]
        # target_len = grid_seq_emb.shape[1] - 1
        # if prompt_len == target_len:
        #     grid_seq_emb[:, 1:, :] += projected_prompt_emb
        # elif prompt_len > target_len:
        #     grid_seq_emb[:, 1:, :] += projected_prompt_emb[:, :target_len, :]
        # else:
        #     pad_len = target_len - prompt_len
        #     pad_tensor = torch.zeros(B, pad_len, projected_prompt_emb.size(2), device=projected_prompt_emb.device)
        #     grid_seq_emb[:, 1:, :] += torch.cat([projected_prompt_emb, pad_tensor], dim=1)
        #
        # # ==== Transformer + contrastive loss ====
        # grid_padding_mask = grid_traj > 0
        # grid_seq_emb = self.grid_enc(grid_seq_emb, grid_padding_mask)
        # grid_seq_emb = self.grid_cl_linear(grid_seq_emb)
        # grid_traj_emb = grid_seq_emb[:, 0]
        #
        # road_e = F.normalize(road_traj_emb, dim=-1)
        # grid_e = F.normalize(grid_traj_emb, dim=-1)
        # logits = torch.matmul(grid_e, road_e.T) / self.temp
        #
        # labels = torch.arange(logits.shape[0], device=logits.device)
        # loss_i = F.cross_entropy(logits, labels)
        # loss_t = F.cross_entropy(logits.T, labels)
        # cl_loss = (loss_i + loss_t) / 2
        #
        # # ==== MLM loss ====
        # # === 新的mask逻辑 ===
        # mask_positions, original_tokens = create_unified_mask(road_traj, mask_ratio=0.15)
        # # 存储mask位置供训练代码使用（可选）
        # self.aux = {'mask_pos': mask_positions}
        # # 创建masked轨迹
        # mask_road_traj = road_traj.clone()
        # mask_road_traj[:, mask_positions] = Config.road_special_tokens['mask_token']
        #
        # # 后续处理保持不变
        # mask_road_seq_emb = g_emb[mask_road_traj] + road_weeks_emb + road_minutes_emb
        # # mask_road_seq_emb = self.pe(mask_road_seq_emb)
        # mask_road_seq_emb = self.pe(mask_road_seq_emb, padding_mask=road_padding_mask)
        # mask_road_seq_emb = self.road_enc(
        #     src=mask_road_seq_emb,
        #     type_src=road_type_emb,
        #     mask=road_padding_mask,
        # )
        # mask_road_seq_emb = self.road_cl_linear(mask_road_seq_emb)
        #
        # fusion_seq_emb = self.inter_layer(
        #     src=mask_road_seq_emb,
        #     src_mask=road_padding_mask,
        #     memory=grid_seq_emb,
        #     memory_mask=grid_padding_mask
        # )
        #
        # # === MLM预测和损失计算 ===
        # mlm_prediction = fusion_seq_emb[:, mask_positions]  # [batch_size, num_mask, embed_dim]
        # mlm_prediction = mlm_prediction.reshape(-1, mlm_prediction.size(-1))  # [batch_size*num_mask, embed_dim]
        # mlm_prediction = self.road_mlm_linear(mlm_prediction)
        #
        # # MLM loss
        # mlm_loss = F.cross_entropy(mlm_prediction, original_tokens)
        #
        # return cl_loss, mlm_loss, mlm_prediction

    def tte_test(self, grid_data, road_data):
        # ==== Road encoder ====
        g_input_feature, g_edge_index = road_data['g_input_feature'], road_data['g_edge_index']
        road_traj = road_data['road_traj']
        road_weeks, road_minutes = road_data['road_weeks'], road_data['road_minutes']
        road_type = road_data['road_type']

        road_weeks_emb = self.week_emb_layer(road_weeks)
        road_minutes_emb = self.minute_emb_layer(road_minutes)
        g_emb = self.road_emb_layer(g_input_feature, g_edge_index)
        g_emb = torch.vstack([self.road_padding_token, self.road_cls_token, self.road_mask_token, g_emb])
        road_seq_emb = g_emb[road_traj] + road_weeks_emb + road_minutes_emb
        road_seq_emb = self.pe(road_seq_emb)
        road_type_emb = self.pe(self.type_emb_layer(road_type))
        road_padding_mask = road_traj > 0
        road_seq_emb = self.road_enc(road_seq_emb, road_type_emb, road_padding_mask)
        road_seq_emb = self.road_cl_linear(road_seq_emb)

        # ==== Grid encoder ====
        grid_image = grid_data['grid_image']
        grid_traj = grid_data['grid_traj']
        grid_time_emb = grid_data['grid_time_emb']
        grid_feature = grid_data['grid_feature']

        if grid_image.dim() == 3:
            grid_image = grid_image.unsqueeze(0)
        grid_image = grid_image.permute(0, 3, 1, 2)
        grid_image = self.transform(grid_image)

        # cnn_feat = self.grid_conv(grid_image)
        # B, D, H, W = cnn_feat.shape
        # cnn_feat_flat = cnn_feat.permute(0, 2, 3, 1).reshape(B * H * W, D)
        # grid_image: (B, C, H, W)
        # B, C, H, W = grid_image.shape
        #
        # # 构图 + 准备 GAT 输入
        # grid_graph_batch = self.grid_to_graph(grid_image)  # grid_image 必须是 [B, C, H, W]
        #
        # # 将 grid_image 转换为节点特征
        # # 先转为 (B, H, W, C)
        # grid_image_flat = grid_image.permute(0, 2, 3, 1).reshape(B * H * W, C)  # (B*H*W, C)
        #
        # # 投影成 GAT 输入维度
        # x = self.grid_raw_proj(grid_image_flat)  # (B*H*W, gat_input_dim)
        #
        # # GAT 编码
        # edge_index = grid_graph_batch.edge_index
        # batch = grid_graph_batch.batch
        # grid_node_emb = self.grid_gat(x, edge_index, batch)
        # grid_node_emb = self.grid_gat_proj_to_transformer(grid_node_emb)

        # grid_graph_batch = self.grid_to_graph(grid_image)
        # grid_node_emb = self.grid_gat(cnn_feat_flat, grid_graph_batch.edge_index)
        # grid_node_emb = self.grid_gat_proj_to_transformer(grid_node_emb)

        #####
        B, C, H, W = grid_image.shape
        grid_graph_batch = self.grid_to_graph(grid_image)
        grid_image_flat = grid_image.permute(0, 2, 3, 1).reshape(B * H * W, C)
        grid_feat_flat = self.grid_raw_proj(grid_image_flat)
        grid_node_emb = self.grid_gat(grid_feat_flat, grid_graph_batch.edge_index)
        grid_node_emb = self.grid_gat_proj_to_transformer(grid_node_emb)
        ######
        full_grid_emb = torch.cat([
            self.grid_padding_token.unsqueeze(0),
            self.grid_cls_token.unsqueeze(0),
            grid_node_emb
        ], dim=0)

        grid_seq_emb_flat = full_grid_emb[grid_traj]
        grid_seq_emb_combined = torch.cat([grid_seq_emb_flat, grid_feature], dim=-1)
        grid_seq_emb = self.fusion_linear(grid_seq_emb_combined) + grid_time_emb
        grid_seq_emb = self.pe(grid_seq_emb)

        # ==== Prompt 融合 ====
        B = grid_traj.shape[0]
        batch_index = torch.arange(B, device=grid_traj.device)
        file_path = f"./model/Embeddings/porto/train.h5"
        prompt_emb = self.load_prompt_embedding_from_h5(file_path, device=grid_seq_emb.device, indices=batch_index)
        projected_prompt_emb = self.prompt_fusion_proj(prompt_emb)
        prompt_len = projected_prompt_emb.shape[1]
        target_len = grid_seq_emb.shape[1] - 1
        if prompt_len == target_len:
            grid_seq_emb[:, 1:, :] += projected_prompt_emb
        elif prompt_len > target_len:
            grid_seq_emb[:, 1:, :] += projected_prompt_emb[:, :target_len, :]
        else:
            pad_len = target_len - prompt_len
            pad_tensor = torch.zeros(B, pad_len, projected_prompt_emb.size(2), device=grid_seq_emb.device)
            grid_seq_emb[:, 1:, :] += torch.cat([projected_prompt_emb, pad_tensor], dim=1)

        grid_padding_mask = grid_traj > 0
        grid_seq_emb = self.grid_enc(grid_seq_emb, grid_padding_mask)
        grid_seq_emb = self.grid_cl_linear(grid_seq_emb)

        # ==== Fusion ====
        fusion_seq_emb = self.inter_layer(
            src=road_seq_emb,
            src_mask=road_padding_mask,
            memory=grid_seq_emb,
            memory_mask=grid_padding_mask
        )
        fusion_traj_emb = fusion_seq_emb[:, 0]
        return fusion_traj_emb

