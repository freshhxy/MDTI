import pandas as pd
import numpy as np
import os

def preprocess_and_save_npz(traj_csv_path, index_dir, save_dir, seq_len=12):
    os.makedirs(save_dir, exist_ok=True)

    # 读取轨迹数据
    traj_df = pd.read_csv(traj_csv_path)  # shape: (N, D)
    traj_array = traj_df.values           # ndarray

    for split in ['train', 'eval', 'test']:
        index_path = os.path.join(index_dir, f"{split}_index.npy")
        indices = np.load(index_path, allow_pickle=True)

        x_list, y_list = [], []

        for idx in indices:
            idx = int(idx)
            if idx + 2 * seq_len > len(traj_array):
                continue
            x = traj_array[idx:idx + seq_len]      # (seq_len, D)
            y = traj_array[idx + seq_len:idx + 2*seq_len]

            # 增加维度为模型输入格式 (B, T, N, F) 做准备：我们这里 N=1，先占位
            x = np.expand_dims(x, axis=1)   # (seq_len, 1, F)
            y = np.expand_dims(y, axis=1)

            x_list.append(x)
            y_list.append(y)

        x_array = np.array(x_list)  # (B, seq_len, 1, F)
        y_array = np.array(y_list)

        np.savez(os.path.join(save_dir, f"{split}.npz"), x=x_array, y=y_array)
        print(f"[✓] Saved {split}.npz, shape: x {x_array.shape}, y {y_array.shape}")

# === 使用示例 ===
if __name__ == "__main__":
    preprocess_and_save_npz(
        traj_csv_path="./data/xian/traj.csv",
        index_dir="./data/xian",
        save_dir="./data/xian/rn/traj",   # 会生成 train/val/test.npz 到这里
        seq_len=12
    )
