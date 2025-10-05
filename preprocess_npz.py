import os
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
import gc
from dataset.grid_space import GridSpace  # 假设你已有此模块

def safe_eval_gps_string(gps_str):
    try:
        return ast.literal_eval(gps_str)
    except:
        return []

def process_split_memmap(split, df_chunk, indices, gs, grid_embedding, seq_len, emb_dim, save_path):
    x_list, mask_list, y_list = [], [], []

    for idx in tqdm(indices, desc=f"[{split}]"):
        gps = safe_eval_gps_string(df_chunk.at[idx, 'gps'])
        if len(gps) < seq_len * 2:
            continue

        x_ids = [gs.get_gridid_by_point(lon, lat) for lon, lat in gps[:seq_len]]
        y_ids = [gs.get_gridid_by_point(lon, lat) for lon, lat in gps[seq_len:seq_len*2]]
        unique = sorted(set(x_ids))
        id2i = {g: i for i, g in enumerate(unique)}
        N = len(unique)

        x_t = np.zeros((seq_len, N, emb_dim), dtype=np.float32)
        x_m = np.zeros((seq_len, N), dtype=bool)
        for t, gid in enumerate(x_ids):
            x_t[t, id2i[gid]] = grid_embedding[gid]
            x_m[t, id2i[gid]] = True

        y_t = np.stack([grid_embedding[gid] for gid in y_ids], axis=0)

        x_list.append(x_t)
        mask_list.append(x_m)
        y_list.append(y_t)

    if not x_list:
        print(f"[!] {split} 无有效轨迹，跳过保存")
        return

    N_max = max(mat.shape[1] for mat in x_list)
    B = len(x_list)

    x_mm = np.memmap(save_path + '_x.dat', dtype='float32', mode='w+', shape=(B, seq_len, N_max, emb_dim))
    m_mm = np.memmap(save_path + '_m.dat', dtype='bool', mode='w+', shape=(B, seq_len, N_max))
    y_mm = np.memmap(save_path + '_y.dat', dtype='float32', mode='w+', shape=(B, seq_len, emb_dim))

    for i in range(B):
        x_pad = np.pad(x_list[i], ((0,0), (0,N_max - x_list[i].shape[1]), (0,0)), 'constant')
        m_pad = np.pad(mask_list[i], ((0,0), (0,N_max - mask_list[i].shape[1])), 'constant')
        x_mm[i], m_mm[i], y_mm[i] = x_pad, m_pad, y_list[i]

    x_mm.flush()
    m_mm.flush()
    y_mm.flush()

    np.savez(
        save_path + '.npz',
        x_file=save_path + '_x.dat',
        x_mask_file=save_path + '_m.dat',
        y_file=save_path + '_y.dat',
        shape=(B, seq_len, N_max, emb_dim)
    )
    print(f"[✓] 保存 {split}.npz → {save_path}.npz，B={B}, N_max={N_max}")
    del x_mm, m_mm, y_mm
    gc.collect()

def main():
    traj_csv = "./data/porto/traj.csv"
    index_dir = "./data/porto"
    save_dir = "./data/porto_bt_nf_npz_memmap"
    os.makedirs(save_dir, exist_ok=True)

    seq_len, emb_dim = 12, 768
    df = pd.read_csv(traj_csv)
    gs = GridSpace(100, 100, -8.75, 41.1, -8.5, 41.3)
    embedding = np.random.randn(gs.grid_num, emb_dim).astype(np.float32)

    for split in ['train', 'eval', 'test']:
        idx_path = os.path.join(index_dir, f"{split}_index.npy")
        if not os.path.exists(idx_path):
            print(f"[!] 缺失索引文件: {idx_path}，跳过 {split}")
            continue
        indices = np.load(idx_path, allow_pickle=True)
        save_path = os.path.join(save_dir, split)
        process_split_memmap(split, df, indices, gs, embedding, seq_len, emb_dim, save_path)

if __name__ == "__main__":
    main()
