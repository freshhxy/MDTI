import pandas as pd
import numpy as np
import os
import ast
from tqdm import tqdm
import time
import psutil
from dataset.grid_space import GridSpace

def safe_eval_gps_string(gps_str):
    try:
        return ast.literal_eval(gps_str)
    except:
        return []

def get_gridid(lon, lat, gs):
    return gs.get_gridid_by_point(lon, lat)

def preprocess_streaming(traj_csv_path, index_dir, save_dir, gs, grid_embedding, seq_len=12, min_free_mem_gb=20):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(traj_csv_path)
    emb_dim = grid_embedding.shape[1]

    for split in ['train', 'eval', 'test']:
        index_path = os.path.join(index_dir, f"{split}_index.npy")
        if not os.path.exists(index_path):
            print(f"[!] 缺失索引: {index_path}，跳过 {split}")
            continue

        indices = np.load(index_path, allow_pickle=True)
        save_split = os.path.join(save_dir, f"{split}")
        os.makedirs(save_split, exist_ok=True)

        sample_count = 0
        for i, idx in enumerate(tqdm(indices, desc=f"[{split}]")):
            # 检查内存占用：只有当“已使用”内存小于阈值才运行
            used_mem_gb = psutil.virtual_memory().used / 1024 ** 3
            if used_mem_gb > (188 - min_free_mem_gb):  # 你总共内存约 188G
                print(f"[WAIT] 当前内存占用 {used_mem_gb:.2f} GB，等待释放...")
                while psutil.virtual_memory().used / 1024 ** 3 > (188 - min_free_mem_gb):
                    time.sleep(10)

            gps = safe_eval_gps_string(df.at[int(idx), 'gps'])
            if len(gps) < 2 * seq_len:
                continue

            gps_x = gps[:seq_len]
            gps_y = gps[seq_len:2 * seq_len]

            x_ids = [get_gridid(lon, lat, gs) for lon, lat in gps_x]
            y_ids = [get_gridid(lon, lat, gs) for lon, lat in gps_y]

            unique_ids = sorted(set(x_ids + y_ids))
            id2idx = {gid: i for i, gid in enumerate(unique_ids)}
            N = len(unique_ids)

            x_tensor = np.zeros((seq_len, N, emb_dim), dtype=np.float32)
            y_tensor = np.zeros((seq_len, N, emb_dim), dtype=np.float32)

            for t, gid in enumerate(x_ids):
                x_tensor[t, id2idx[gid]] = grid_embedding[gid]
            for t, gid in enumerate(y_ids):
                y_tensor[t, id2idx[gid]] = grid_embedding[gid]

            np.savez(os.path.join(save_split, f"{i}.npz"), x=x_tensor, y=y_tensor)
            sample_count += 1

        print(f"[✓] 已保存 {sample_count} 条样本至 {save_split}/")

if __name__ == "__main__":
    traj_csv_path = "./data/porto/traj.csv"
    index_dir = "./data/porto"
    save_dir = "./data/porto/split_npz"
    seq_len = 12
    emb_dim = 768

    gs = GridSpace(100, 100, -8.75, 41.1, -8.5, 41.3)
    grid_embedding = np.random.randn(gs.grid_num, emb_dim).astype(np.float32)

    preprocess_streaming(traj_csv_path, index_dir, save_dir, gs, grid_embedding, seq_len, min_free_mem_gb=20)
