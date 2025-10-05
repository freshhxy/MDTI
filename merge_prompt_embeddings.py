import os
import h5py
import numpy as np
from tqdm import tqdm


def merge_h5_files(input_dir, output_path, total_files):
    merged_embeddings = []

    for i in tqdm(range(total_files), desc="Merging"):
        file_path = os.path.join(input_dir, f"train_{i}.h5")
        with h5py.File(file_path, 'r') as f:
            emb = f['embeddings'][:]  # shape: (1, N, 768)
            merged_embeddings.append(emb[0])  # (N, 768)

    merged_embeddings = np.stack(merged_embeddings)  # shape: (total_files, N, 768)

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('embeddings', data=merged_embeddings, compression="gzip")

    print(f"Merged {total_files} files into {output_path}. Final shape: {merged_embeddings.shape}")


if __name__ == "__main__":
    input_dir = "./xian/Embeddings/xian/train"
    output_path = "./model/Embeddings/xian/train.h5"
    total_files = 1874
    merge_h5_files(input_dir, output_path, total_files)
