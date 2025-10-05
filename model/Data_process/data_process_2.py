# import numpy as np
# import os
# import ast  # 导入 ast 模块，用于安全地评估字符串
# import h5py
#
# # --- 辅助函数：将字符串形式的列表转换为数值数组 ---
# def _convert_string_lists_in_array(input_numpy_array):
#     """
#     遍历 NumPy 数组，将其中形如 '[x, y, z]' 的字符串转换为数值型 NumPy 数组。
#     假定所有元素最终都应为数值。
#     """
#     output_list = []
#
#     # 展平数组以确保遍历所有元素
#     flat_array = input_numpy_array.flatten()
#
#     for item in flat_array:
#         if isinstance(item, str) and item.startswith('[') and item.endswith(']'):
#             try:
#                 # 使用 ast.literal_eval 安全地将字符串评估为 Python 列表或元组
#                 evaluated_item = ast.literal_eval(item)
#                 # 将 Python 列表/元组转换为 NumPy 数组并指定 dtype
#                 if not evaluated_item:  # 处理空列表的情况
#                     output_list.append(np.array([], dtype=np.float32))
#                 else:
#                     output_list.append(np.asarray(evaluated_item, dtype=np.float32))
#             except (ValueError, SyntaxError) as e:
#                 # 警告：如果字符串不是有效的列表格式，尝试直接转换为浮点数
#                 print(f"警告：无法将字符串 '{item}' 解析为列表。错误：{e}。尝试直接转换为浮点数。")
#                 try:
#                     output_list.append(np.float32(item))
#                 except ValueError:
#                     # 如果直接转换也失败，则抛出错误
#                     raise ValueError(f"无法将 '{item}' 转换为浮点数或列表。请检查数据格式。")
#         else:
#             # 对于非字符串或不是列表字符串的元素，尝试直接转换为浮点数
#             try:
#                 output_list.append(np.float32(item))
#             except ValueError:
#                 raise ValueError(f"无法将 '{item}' 转换为浮点数。请检查数据格式。")
#
#     # 尝试将处理后的列表堆叠成一个 NumPy 数组。
#     # 这假设所有处理后的“元素”（例如，一个模式的展平向量）长度相同。
#     # 如果长度不同，np.stack 会失败，这可能意味着您的数据结构需要填充或其他处理。
#     try:
#         print(f"[DEBUG] 每个清洗后元素 shape1: {[arr.shape for arr in output_list]}")
#         return np.stack(output_list)
#     except ValueError as e:
#         print(f"警告：堆叠处理后的模式数组时失败，可能是由于长度不一致。错误：{e}")
#         # 如果堆叠失败，回退到对象数组。但请注意，`euclidean_distances` 通常要求数值数组。
#         print(f"[DEBUG] 每个清洗后元素 shape2: {[arr.shape for arr in output_list]}")
#         return np.asarray(output_list, dtype=object)
#
#
#
# # --- DataLoader 类定义 ---
# class DataLoader:
#     def __init__(self, xs, ys, batch_size, scaler=None, shuffle=False):
#         self.xs = xs  # h5py.Dataset
#         self.ys = ys
#         self.batch_size = batch_size
#         self.size = len(xs)
#         self.num_batch = int(np.ceil(self.size / self.batch_size))
#         self.current_ind = 0
#         self.scaler = scaler
#         self.shuffle = shuffle
#
#         self.indices = np.arange(self.size)
#         if shuffle:
#             np.random.shuffle(self.indices)
#
#     def get_iterator(self):
#         self.current_ind = 0
#
#         def _wrapper():
#             while self.current_ind < self.num_batch:
#                 start_ind = self.current_ind * self.batch_size
#                 end_ind = min((self.current_ind + 1) * self.batch_size, self.size)
#                 batch_indices = self.indices[start_ind:end_ind]
#
#                 x_batch = self.xs[batch_indices]  # 懒加载
#                 y_batch = self.ys[batch_indices]
#
#                 # 只对通道 1 做归一化（例如：速度）
#                 if self.scaler is not None:
#                     x_batch[..., 1] = self.scaler.transform(x_batch[..., 1])
#                     y_batch[..., 1] = self.scaler.transform(y_batch[..., 1])
#
#                 yield x_batch, y_batch
#                 self.current_ind += 1
#
#         return _wrapper()
#
#
#
# # --- StandardScaler 类定义 ---
# class StandardScaler:
#     """
#     Standard the input
#     """
#
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std
#
#     def transform(self, data):
#         return (data - self.mean) / self.std
#
#     def inverse_transform(self, data):
#         return (data * self.std) + self.mean
#
#
# # --- load_dataset 函数定义 ---
# def load_dataset(dataset_dir, dataset_name, seq_len, batch_size):
#     data = {}
#
#     for category in ['train', 'eval', 'test']:
#         file_path = os.path.join(dataset_dir, dataset_name, category + '.h5')
#         f = h5py.File(file_path, 'r')  #  注意：不要在 with 语句中关闭它
#
#         # 存下数据集对象（不是数据）
#         data['x_' + category] = f['x']
#         data['y_' + category] = f['y']
#
#     # 创建 scaler，只缩放第 1 通道
#     scaler = StandardScaler(
#         mean=np.mean(data['x_train'][..., 1]),
#         std=np.std(data['x_train'][..., 1])
#     )
#
#     # 使用 transform 不复制数据，仅在 batch 中做也行
#     data['scaler'] = scaler
#
#     # 构造延迟 DataLoader
#     data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, scaler, shuffle=False)
#     data['eval_loader']   = DataLoader(data['x_eval'], data['y_eval'], batch_size, scaler, shuffle=False)
#     data['test_loader']  = DataLoader(data['x_test'], data['y_test'], batch_size, scaler, shuffle=False)
#
#     # 加载 pattern_keys（仍按原方式即可）
#     pattern_keys = np.load(os.path.join(dataset_dir, dataset_name, f'pattern_keys_{dataset_name}.npy'), allow_pickle=True)
#     #新加的
#
#     return data, pattern_keys
import numpy as np, os, h5py, ast

# —— 工具：打印文件结构，定位真实数据位置 ——
def _inspect_h5(h5file):
    def show(name, obj):
        shape = getattr(obj, "shape", "")
        dtype = getattr(obj, "dtype", "")
        print(f"[H5] {name} | {type(obj).__name__} | shape={shape} | dtype={dtype}")
    h5file.visititems(show)

# —— 解析成真正的 Dataset；支持标量引用/路径/包装组 ——
def _resolve_dataset(h5obj, key):
    obj = h5obj[key]

    # 1) 直接是 Dataset
    if isinstance(obj, h5py.Dataset):
        # 1a) 标量 Dataset 的特殊处理
        if obj.shape == ():
            # 可能是 HDF5 对象引用
            if obj.dtype == h5py.ref_dtype:
                ref = obj[()]
                tgt = h5obj.file[ref]
                if isinstance(tgt, h5py.Dataset):
                    return tgt
                if isinstance(tgt, h5py.Group):
                    # 优先常见子键
                    for cand in ("x","data","values","dataset","array"):
                        if cand in tgt and isinstance(tgt[cand], h5py.Dataset):
                            return tgt[cand]
                    # 回退：取第一个 Dataset
                    for _, v in tgt.items():
                        if isinstance(v, h5py.Dataset):
                            return v
                raise TypeError("标量引用未指向 Dataset")
            # 可能是存放路径的标量字符串
            val = obj[()]
            if isinstance(val, (bytes, str)):
                path = val.decode() if isinstance(val, bytes) else val
                if path in h5obj.file:
                    tgt = h5obj.file[path]
                    if isinstance(tgt, h5py.Dataset):
                        return tgt
                    if isinstance(tgt, h5py.Group):
                        for cand in ("x","data","values","dataset","array"):
                            if cand in tgt and isinstance(tgt[cand], h5py.Dataset):
                                return tgt[cand]
                        for _, v in tgt.items():
                            if isinstance(v, h5py.Dataset):
                                return v
                # 路径无效，继续下面逻辑
        # 非标量 Dataset，直接用
        return obj

    # 2) 是 Group：在常见子键里找 Dataset
    if isinstance(obj, h5py.Group):
        for cand in ("x","data","values","dataset","array"):
            if cand in obj and isinstance(obj[cand], h5py.Dataset):
                return obj[cand]
        for _, v in obj.items():
            if isinstance(v, h5py.Dataset):
                return v

    # 3) 都没找到：打印结构并报错
    print("[DEBUG] 未能解析到 Dataset。文件结构如下：")
    _inspect_h5(h5obj.file)
    raise TypeError(f"键 '{key}' 无法解析为 Dataset，实际类型：{type(obj)}")

# —— 安全裁剪时间维（仅当存在第1维） ——
def _maybe_time_slice(ds, seq_len):
    if seq_len is None: return ds
    if hasattr(ds, "ndim") and ds.ndim >= 2:
        return ds[:, :min(seq_len, ds.shape[1]), ...]
    return ds

# —— 按块统计指定通道的 mean/std（要求末维是通道） ——
def _stream_mean_std_channel(ds, channel_idx, seq_len=None, chunk=1024):
    if ds.ndim < 3 or ds.shape[-1] <= channel_idx:
        raise ValueError(f"x.shape={ds.shape} 不含通道 {channel_idx}，请确认布局应为 (..., C)")
    n = ds.shape[0]
    t = ds.shape[1] if seq_len is None else min(seq_len, ds.shape[1])
    count = 0; mean = 0.0; M2 = 0.0
    for s in range(0, n, chunk):
        e = min(s+chunk, n)
        x = np.asarray(ds[s:e, :t, ...])[..., channel_idx].reshape(-1)
        if x.size == 0: continue
        cnt = x.size; m = float(x.mean()); m2 = float(((x-m)**2).sum())
        delta = m - mean; tot = count + cnt
        mean = mean + delta * (cnt / tot)
        M2 = M2 + m2 + delta*delta * (count*cnt / tot)
        count = tot
    if count == 0: raise ValueError("统计失败：有效样本为 0")
    std = float(np.sqrt(M2 / count))
    return float(mean), std

class StandardScaler:
    def __init__(self, mean, std): self.mean=float(mean); self.std=float(std)
    def transform(self, x): return (x-self.mean)/self.std if self.std!=0.0 else x*0.0
    def inverse_transform(self, x): return x*self.std + self.mean

class DataLoader:
    def __init__(self, xs, ys, batch, scaler=None, shuffle=False, seq_len=None):
        self.xs, self.ys = xs, ys
        self.batch = int(batch)
        self.size = int(xs.shape[0])
        self.num_batch = int(np.ceil(self.size/self.batch))
        self.scaler, self.shuffle, self.seq_len = scaler, shuffle, seq_len
        self.idx = np.arange(self.size);
        if shuffle: np.random.shuffle(self.idx)
        self.cur = 0
    def get_iterator(self):
        self.cur = 0
        def it():
            while self.cur < self.num_batch:
                s = self.cur*self.batch; e = min((self.cur+1)*self.batch, self.size)
                sel = self.idx[s:e]
                xb = self.xs[sel]; yb = self.ys[sel]
                if self.seq_len is not None and xb.ndim >= 2:
                    t = min(self.seq_len, xb.shape[1])
                    xb = xb[:, :t, ...]; yb = yb[:, :t, ...]
                if self.scaler is not None:
                    xb = np.asarray(xb); yb = np.asarray(yb)
                    if xb.ndim < 3 or xb.shape[-1] <= 1:
                        raise ValueError(f"批次形状不含通道1：{xb.shape}")
                    xb[...,1] = self.scaler.transform(xb[...,1])
                    yb[...,1] = self.scaler.transform(yb[...,1])
                self.cur += 1
                yield xb, yb
        return it()

def load_dataset(dataset_dir, dataset_name, seq_len, batch_size):
    data = {}; data['_files'] = {}
    for cat in ['train','eval','test']:
        fp = os.path.join(dataset_dir, dataset_name, f"{cat}.h5")
        f = h5py.File(fp, 'r'); data['_files'][cat] = f
        x = _resolve_dataset(f, 'x'); y = _resolve_dataset(f, 'y')
        data['x_'+cat] = _maybe_time_slice(x, seq_len)
        data['y_'+cat] = _maybe_time_slice(y, seq_len)

    # # 计算通道1统计量；若失败，打印结构并抛清晰错误
    # try:
    #     mean, std = _stream_mean_std_channel(data['x_train'], channel_idx=1, seq_len=seq_len, chunk=1024)
    # except Exception as e:
    #     print("[DEBUG] 训练文件结构：")
    #     _inspect_h5(data['_files']['train'])
    #     raise

    scaler = StandardScaler(mean=data['x_train'][..., 1].mean(), std=data['x_train'][..., 1].std())
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, scaler=scaler, shuffle=False, seq_len=seq_len)
    data['eval_loader']  = DataLoader(data['x_eval'],  data['y_eval'],  batch_size, scaler=scaler, shuffle=False, seq_len=seq_len)
    data['test_loader']  = DataLoader(data['x_test'],  data['y_test'],  batch_size, scaler=scaler, shuffle=False, seq_len=seq_len)

    pattern_keys = np.load(os.path.join(dataset_dir, dataset_name, f"pattern_keys_{dataset_name}.npy"), allow_pickle=True)
    return data, pattern_keys
