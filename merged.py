import os
import numpy as np
from tqdm import tqdm
import shutil
import h5py
def merge_npz_in_batches(npz_dir, output_path, batch_size=500, only_merge=False):
    tmp_dir = os.path.join(npz_dir, "__tmp_merge__")
    os.makedirs(tmp_dir, exist_ok=True)

    if not only_merge:
        files = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
        print(f"共发现 {len(files)} 个 .npz 文件，开始分批合并...")

        seq_len, emb_dim = None, None
        max_N = 0
        part_idx = 0
        buffer_x, buffer_y = [], []

        for i, fname in enumerate(tqdm(files, desc="扫描并分批处理")):
            try:
                data = np.load(os.path.join(npz_dir, fname))
                if 'x' not in data or 'y' not in data:
                    continue
                x, y = data['x'], data['y']

                if x.ndim == 3:
                    x = np.expand_dims(x, axis=0)
                if y.ndim == 3:
                    y = np.expand_dims(y, axis=0)

                if x.ndim != 4 or y.ndim != 4:
                    print(f"[!] 跳过 {fname}：维度不合法 {x.shape}")
                    continue
            except Exception as e:
                print(f"[!] 跳过 {fname}：读取失败 → {e}")
                continue

            if seq_len is None:
                seq_len = x.shape[1]
                emb_dim = x.shape[3]

            max_N = max(max_N, x.shape[2])
            buffer_x.append(x)
            buffer_y.append(y)

            if len(buffer_x) >= batch_size:
                save_batch(buffer_x, buffer_y, tmp_dir, part_idx, seq_len, max_N, emb_dim)
                buffer_x, buffer_y = [], []
                part_idx += 1

        if buffer_x:
            save_batch(buffer_x, buffer_y, tmp_dir, part_idx, seq_len, max_N, emb_dim)
            part_idx += 1
    else:
        # 自动统计已有多少中间文件
        part_idx = len([f for f in os.listdir(tmp_dir) if f.startswith("x_part_")])

    import h5py

    # ==================== 最终合并 ====================
    print("[+] 开始最终拼接所有中间部分（流式）...")
    total_size = 0
    sample_shape = None
    T, F = None, None
    max_N = 0

    # 第一次遍历：确定数据总大小
    for i in tqdm(range(part_idx), desc="统计总大小和最大 N"):
        x_part = np.load(os.path.join(tmp_dir, f"x_part_{i}.npy"))
        total_size += x_part.shape[0]
        T = x_part.shape[1]
        F = x_part.shape[3]
        max_N = max(max_N, x_part.shape[2])  # 更新最大 N

    # 创建 h5 文件进行流式写入
    h5_path = output_path.replace(".npz", ".h5")
    with h5py.File(h5_path, "w") as f:
        dset_x = f.create_dataset("x", shape=(total_size, T, max_N, F), dtype='float32')
        dset_y = f.create_dataset("y", shape=(total_size, T, max_N, F), dtype='float32')

        offset = 0
        for i in tqdm(range(part_idx), desc="写入到 h5"):
            x_part = np.load(os.path.join(tmp_dir, f"x_part_{i}.npy"))
            y_part = np.load(os.path.join(tmp_dir, f"y_part_{i}.npy"))
            bsz = x_part.shape[0]
            # 自动右侧 padding 补齐维度
            if x_part.shape[2] < max_N:
                pad_width = ((0, 0), (0, 0), (0, max_N - x_part.shape[2]), (0, 0))
                x_part = np.pad(x_part, pad_width, mode='constant')
                y_part = np.pad(y_part, pad_width, mode='constant')

            dset_x[offset:offset + bsz] = x_part
            dset_y[offset:offset + bsz] = y_part
            offset += bsz

        print(f"[✓] 合并完成: {h5_path}，shape: x {dset_x.shape}, y {dset_y.shape}")
    shutil.rmtree(tmp_dir)


def save_batch(x_list, y_list, tmp_dir, idx, seq_len, max_N, emb_dim):
    B = sum(x.shape[0] for x in x_list)
    x_batch = np.zeros((B, seq_len, max_N, emb_dim), dtype=np.float32)
    y_batch = np.zeros((B, seq_len, max_N, emb_dim), dtype=np.float32)

    offset = 0
    for x, y in zip(x_list, y_list):
        bsz = x.shape[0]
        x_batch[offset:offset+bsz, :, :x.shape[2], :] = x
        y_batch[offset:offset+bsz, :, :y.shape[2], :] = y
        offset += bsz

    np.save(os.path.join(tmp_dir, f"x_part_{idx}.npy"), x_batch)
    np.save(os.path.join(tmp_dir, f"y_part_{idx}.npy"), y_batch)


if __name__ == "__main__":
    merge_npz_in_batches(
        npz_dir="./split_npz/test",  # 输入目录（多个小 npz）
        output_path="./test.h5",    # 合并后保存路径
        batch_size=32, # 每500个文件暂存一次，可调小避免OOM
        only_merge = False  # 只执行最终拼接
    )
