import random
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from config.config import Config



class TrajDataset(Dataset):
    def __init__(self, traj_data):
        self.traj_data = traj_data

    def __len__(self):
        return len(self.traj_data)

    def __getitem__(self, idx):
        row = self.traj_data.iloc[idx]
        return {
            "road_traj": row["road_traj"],
            "grid_traj": row["grid_traj"],
            "ptime": row["ptime"],
            "grid_feature": row["grid_feature"],
            "time_emb": row["time_emb"],
            "road_type": row["road_type"]
        }


def continuous_mask_sequence(sequence, mask_percentage, mask_length):
    num_elements = len(sequence)
    num_elements_to_mask = int(num_elements * mask_percentage)
    masked_sequence = sequence.clone()

    num_continuous_intervals = num_elements - mask_length + 1

    num_intervals_to_mask = num_elements_to_mask // mask_length

    max_interval = num_continuous_intervals - num_intervals_to_mask

    start_indices = set()
    # TODO: An infinite loop could happen here, but this rarely happens because the masking rate is too small,
    #       and we can exit by setting the number of loops
    # num = 0
    while len(start_indices) < num_intervals_to_mask:
        start_idx = random.randint(0, max_interval)
        if all(start_idx < s or start_idx > s + mask_length for s in start_indices):
            start_indices.add(start_idx)
        # num += 1

    for start_idx in start_indices:
        masked_sequence[start_idx:start_idx+mask_length] = Config.road_special_tokens['mask_token']

    return masked_sequence


class TrajDataLoader:
    def __init__(self):
        self.batch_size = Config.batch_size
        self.num_workers = 8

    def get_data_loader(self, traj_data, is_shuffle=False,sample_ratio=1.0):
        traj_list = traj_data
        if sample_ratio < 1.0:
            sample_len = int(len(traj_data) * sample_ratio)
            traj_list = traj_data[:sample_len]
            print(f"[DataLoader] Sampled {sample_len} trajectories from {len(traj_list)}")
        dataset = TrajDataset(traj_data=traj_list)

        data_loader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 shuffle=is_shuffle,
                                 num_workers=self.num_workers,
                                 collate_fn=self._collate_func,
                                 drop_last=True  # 加上这一行！
                                 )
        return data_loader

    def _collate_func(self, batch):
        bz = len(batch)
        # road_traj_list = [traj.road_traj for traj in data_df]
        # grid_traj_list = [traj.grid_traj for traj in data_df]
        #
        # road_temporal_list = [traj.ptime for traj in data_df]
        #
        # grid_feature_list = [traj.grid_feature for traj in data_df]
        # # raw_grid_data_for_prompt_list = [traj.raw_grid_data for traj in data_df]
        # raw_grid_data_for_prompt_list = [traj["grid_feature"] for traj in data_df]
        road_traj_list = [item["road_traj"] for item in batch]
        grid_traj_list = [item["grid_traj"] for item in batch]
        road_temporal_list = [item["ptime"] for item in batch]
        grid_feature_list = [np.array(item["grid_feature"]) for item in batch]
        raw_grid_data_for_prompt_list = [torch.tensor(item["grid_feature"], dtype=torch.float32) for item in batch]
        # print(f"[DEBUG] len(raw_grid_data_for_prompt_list): {len(raw_grid_data_for_prompt_list)}")
        # print(f"[DEBUG] type: {type(raw_grid_data_for_prompt_list[0])}")
        # print(f"[DEBUG] shape: {getattr(raw_grid_data_for_prompt_list[0], 'shape', 'no shape')}")
        MAX_PROMPT_SEQ_LEN = 512
        # max_prompt_seq_len = max(data.shape[0] for data in raw_grid_data_for_prompt_list)
        # 限制最大序列长度
        max_prompt_seq_len = min(max(data.shape[0] for data in raw_grid_data_for_prompt_list), MAX_PROMPT_SEQ_LEN)

        num_grid_nodes = raw_grid_data_for_prompt_list[0].shape[1]
        raw_prompt_dim = raw_grid_data_for_prompt_list[0].shape[1]
        road_lens = [len(path) for path in road_traj_list]
        grid_lens = [len(grid) for grid in grid_traj_list]
        max_road_len = max(road_lens)
        max_grid_len = max(grid_lens)

        road_traj_inputs = torch.zeros((bz, max_road_len + 1), dtype=torch.long)
        grid_traj_inputs = torch.zeros((bz, max_grid_len + 1), dtype=torch.long)
        mask_road_traj_inputs = torch.zeros((bz, max_road_len + 1), dtype=torch.long)
        road_type_inputs = torch.zeros((bz, max_road_len + 1), dtype=torch.long)
        grid_feature_inputs = torch.zeros((bz, max_grid_len + 1, 4), dtype=torch.float32)
        grid_time_emb_inputs = torch.zeros((bz, max_grid_len + 1, Config.hidden_emb_dim), dtype=torch.float32)
        road_week_inputs = torch.zeros((bz, max_road_len + 1), dtype=torch.long)
        road_minute_inputs = torch.zeros((bz, max_road_len + 1), dtype=torch.long)
        raw_grid_data_for_prompt_inputs = torch.zeros(
            (bz, max_prompt_seq_len, num_grid_nodes, raw_prompt_dim), dtype=torch.float32
        )
        # road_lens = [len(path) for path in road_traj_list]
        # grid_lens = [len(grid) for grid in grid_traj_list]
        # max_road_len = max(road_lens)
        # max_grid_len = max(grid_lens)
        #
        # max_prompt_seq_len = max([data.shape[0] for data in raw_grid_data_for_prompt_list])
        # num_grid_nodes = raw_grid_data_for_prompt_list[0].shape[1]  # Assuming consistent number of nodes
        # raw_prompt_dim = raw_grid_data_for_prompt_list[0].shape[2]  # Assuming consistent raw dim
        #
        # # road_traj_inputs = torch.zeros((bz, max_road_len + 1), dtype=torch.long)
        # # grid_traj_inputs = torch.zeros((bz, max_grid_len + 1), dtype=torch.long)
        # # mask_road_traj_inputs = torch.zeros((bz, max_road_len + 1), dtype=torch.long)
        # # road_type_inputs = torch.zeros((bz, max_road_len + 1), dtype=torch.long)
        # road_traj_inputs = torch.zeros((bz, max_road_len + 1), dtype=torch.long)
        # grid_traj_inputs = torch.zeros((bz, max_grid_len + 1), dtype=torch.long)
        # mask_road_traj_inputs = torch.zeros((bz, max_road_len + 1), dtype=torch.long)
        # road_type_inputs = torch.zeros((bz, max_road_len + 1), dtype=torch.long)
        #
        # grid_feature_inputs = torch.zeros((bz, max_grid_len + 1, 4), dtype=torch.float32)
        #
        # grid_time_emb_inputs = torch.zeros((bz, max_grid_len + 1, Config.hidden_emb_dim), dtype=torch.float32)
        #
        # road_week_inputs = torch.zeros((bz, max_road_len + 1), dtype=torch.long)
        # road_minute_inputs = torch.zeros((bz, max_road_len + 1), dtype=torch.long)
        # raw_grid_data_for_prompt_inputs = torch.zeros(
        #     (bz, max_prompt_seq_len, num_grid_nodes, raw_prompt_dim), dtype=torch.float32
        # )

        for i in range(bz):
            path = road_traj_list[i]
            road_len = len(path)
            grid = grid_traj_list[i]
            grid_len = len(grid)

            grid_temporal_emb = torch.tensor(batch[i]['time_emb'], dtype=torch.float32)

            road_type = batch[i]['road_type']
            road_type_inputs[i, 1:road_len + 1] = torch.LongTensor(road_type) + 1

            grid_feature_inputs[i, 1:grid_len + 1] = torch.tensor(grid_feature_list[i], dtype=torch.float32)

            road_shift_with_tokens = torch.LongTensor(path) + len(Config.road_special_tokens)
            road_traj_inputs[i, 1:road_len + 1] = road_shift_with_tokens
            road_traj_inputs[i, 0] = Config.road_special_tokens['cls_token']

            mask_road_traj = continuous_mask_sequence(road_shift_with_tokens,
                                                      Config.mask_ratio,
                                                      Config.mask_length)

            mask_road_traj_inputs[i, 1:road_len + 1] = mask_road_traj
            mask_road_traj_inputs[i, 0] = Config.road_special_tokens['cls_token']

            grid_shift_with_tokens = torch.LongTensor(grid) + len(Config.grid_special_tokens)
            grid_traj_inputs[i, 1:grid_len + 1] = grid_shift_with_tokens
            grid_traj_inputs[i, 0] = Config.grid_special_tokens['cls_token']

            grid_time_emb_inputs[i, 0] = grid_temporal_emb[0]
            grid_time_emb_inputs[i, 1:grid_len + 1] = grid_temporal_emb

            road_temporal = road_temporal_list[i]
            road_date = [datetime.fromtimestamp(t) for t in road_temporal]
            road_weeks = [d.weekday() + 1 for d in road_date]
            road_minutes = [d.minute + 1 + d.hour * 60 for d in road_date]

            road_week_inputs[i, 1:road_len + 1] = torch.LongTensor(road_weeks)
            road_week_inputs[i, 0] = road_weeks[0]
            road_minute_inputs[i, 1:road_len + 1] = torch.LongTensor(road_minutes)
            road_minute_inputs[i, 0] = road_minutes[0]

            # current_raw_grid_data = raw_grid_data_for_prompt_list[i]
            # current_raw_grid_data = current_raw_grid_data.unsqueeze(-1)
            # current_prompt_seq_len = current_raw_grid_data.shape[0]
            # raw_grid_data_for_prompt_inputs[i, :current_prompt_seq_len, :, :] = current_raw_grid_data
            current_raw_grid_data = raw_grid_data_for_prompt_list[i]
            current_raw_grid_data = current_raw_grid_data[:max_prompt_seq_len]  # 截断到最大长度

            # 如果缺失最后一个维度（如 dim=raw_prompt_dim），可在此处 unsqueeze
            if current_raw_grid_data.ndim == 2:  # (seq_len, grid_dim)
                current_raw_grid_data = current_raw_grid_data.unsqueeze(-1)  # => (seq_len, grid_dim, 1)

            current_prompt_seq_len = current_raw_grid_data.shape[0]
            raw_grid_data_for_prompt_inputs[i, :current_prompt_seq_len, :, :] = current_raw_grid_data

        mask_road_index = torch.where(mask_road_traj_inputs == Config.road_special_tokens['mask_token'])

        road_data = {
            'road_traj': road_traj_inputs,
            'mask_road_index': mask_road_index,
            'road_type': road_type_inputs,
            'road_weeks': road_week_inputs,
            'road_minutes': road_minute_inputs,
        }

        grid_data = {
            'grid_traj': grid_traj_inputs,
            'grid_feature': grid_feature_inputs,
            'grid_time_emb': grid_time_emb_inputs,
            'raw_grid_data_for_prompt': raw_grid_data_for_prompt_inputs
        }
        return road_data, grid_data
