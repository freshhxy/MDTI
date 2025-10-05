import os
import gc
import time
import h5py
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2Model
from model.Data_process.data_process import load_dataset
from sklearn.metrics.pairwise import euclidean_distances
from multiprocessing import Process, set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='')
    parser.add_argument('--dataset_name', type=str, default='porto')
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpus', type=str, default='5')

    return parser.parse_args()


# --- 模式划分 ---
def split_pattern(input_sequence):
    trimmed = input_sequence[:12]
    return trimmed.reshape(4, 3, -1)


def compute_similarity(input_split_features, pattern):
    num_patterns, pattern_len, _ = input_split_features.shape
    if pattern.ndim == 3: #16,9
        pattern = pattern.reshape(pattern.shape[0], -1)
    num_patterns = input_split_features.shape[0]
    pattern_len = input_split_features.shape[1]
    x_train = pattern
    sims = []
    for i in range(num_patterns):
        ip = input_split_features[i].reshape(1, -1)
        dist = euclidean_distances(ip, x_train)
        sim = np.exp(-dist / (2 * np.sqrt(pattern_len)))
        sims.append(sim[0])
    return np.array(sims)


def compute_probabilities(similarity_scores):
    exp_scores = np.exp(similarity_scores)
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)


def build_prompt(index, values, trend, top_patterns, top_probs, top_cats):
    values_str = ", ".join([f"{v:.2f}" for v in values])
    patterns_str = ", ".join([str(v) for v in top_patterns.tolist()])
    probs_str = ", ".join([f"{v:.4f}" for v in top_probs.tolist()])
    cats_str = ", ".join([str(c) for c in top_cats.tolist()])
    prompt = (
        f"From {int(index[0])} to {int(index[-1])}, the values were {values_str} at each time step. "
        f"The accumulated trend was {trend:.4f}. The most similar historical patterns are {patterns_str}, "
        f"ranked by similarity scores {probs_str} with categories {cats_str}."
    )
    return prompt


def generate_prompt_embeddings(batch_x, pattern_keys, tokenizer, model, device):
    B, T, N, D = batch_x.shape
    prompts = []
    if pattern_keys.ndim == 3:
        if pattern_keys.shape[2] == 1:
            # 从 (16, 3, 1) → (16, 3, 3)
            pattern_keys = np.tile(pattern_keys, (1, 1, 3))
        pattern_keys = pattern_keys.reshape(pattern_keys.shape[0], -1)  # (16, 9)
    elif pattern_keys.ndim == 2 and pattern_keys.shape[1] != 9:
        raise ValueError(f"Expected pattern_keys shape to be (N, 9), got {pattern_keys.shape}")
    for i in range(B):
        for j in range(N):
            traj = batch_x[i, :, j, :]
            index = traj[:, 0]
            values = traj[:, 1]
            trend = np.sum(np.diff(values))

            split_traj = split_pattern(traj)
            sim = compute_similarity(split_traj, pattern_keys)
            prob = compute_probabilities(sim)

            top_cats = np.argsort(prob[0])[::-1][:3]
            top_probs = prob[0][top_cats]
            top_patterns = pattern_keys[top_cats]

            prompt = build_prompt(index, values, trend, top_patterns, top_probs, top_cats)
            prompts.append(prompt)

    tokenized = tokenizer(prompts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    with torch.no_grad():
        output = model(**tokenized).last_hidden_state

    last_token = output[:, -1, :]
    return last_token.view(B, N, -1)


def run_split_on_gpu(split, gpu_id, args):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    tokenizer = GPT2Tokenizer.from_pretrained('./TimeCMA/gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained('./TimeCMA/gpt2').to(device)
    model.eval()
    save_base = f"./porto/Embeddings/{args.dataset_name}/"

    # 子进程内部重新加载数据集（避免 loader/pattern_keys 不能 pickle）
    data_set, pattern_keys = load_dataset(
        args.dataset_dir, args.dataset_name, args.seq_len, args.batch_size
    )
    loader = data_set[f"{split}_loader"]
    data_iter = loader.get_iterator()
    total_batches = loader.num_batch
    save_dir = os.path.join(save_base, split)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[GPU {gpu_id}]开始 `{split}` embedding，共 {total_batches} 个 batch")
    start_time = time.time()

    for i, (x, y) in enumerate(tqdm(data_iter, total=total_batches)):
        x = x[:, :, :, :3]#32，9，24，3
        # print("pattern_keys shape:", pattern_keys.shape)
        embeddings = generate_prompt_embeddings(x, pattern_keys, tokenizer, model, device)

        with h5py.File(os.path.join(save_dir, f"{split}_{i}.h5"), 'w') as hf:
            hf.create_dataset('embeddings', data=embeddings.cpu().numpy())

        del x, y, embeddings
        gc.collect()
        torch.cuda.empty_cache()

    print(f"[GPU {gpu_id}]`{split}` 完成，总耗时 {(time.time() - start_time)/60:.2f} 分钟")


if __name__ == '__main__':
    args = arguments()
    splits = ['train']
    gpu_ids = list(map(int, args.gpus.split(',')))

    processes = []
    for i, split in enumerate(splits):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        p = Process(target=run_split_on_gpu, args=(split, gpu_id, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()