import numpy as np
import os
import ast  # 导入 ast 模块，用于安全地评估字符串


# --- 辅助函数：将字符串形式的列表转换为数值数组 ---
def _convert_string_lists_in_array(input_numpy_array):
    """
    遍历 NumPy 数组，将其中形如 '[x, y, z]' 的字符串转换为数值型 NumPy 数组。
    假定所有元素最终都应为数值。
    """
    output_list = []

    # 展平数组以确保遍历所有元素
    flat_array = input_numpy_array.flatten()

    for item in flat_array:
        if isinstance(item, str) and item.startswith('[') and item.endswith(']'):
            try:
                # 使用 ast.literal_eval 安全地将字符串评估为 Python 列表或元组
                evaluated_item = ast.literal_eval(item)
                # 将 Python 列表/元组转换为 NumPy 数组并指定 dtype
                if not evaluated_item:  # 处理空列表的情况
                    output_list.append(np.array([], dtype=np.float32))
                else:
                    output_list.append(np.asarray(evaluated_item, dtype=np.float32))
            except (ValueError, SyntaxError) as e:
                # 警告：如果字符串不是有效的列表格式，尝试直接转换为浮点数
                print(f"警告：无法将字符串 '{item}' 解析为列表。错误：{e}。尝试直接转换为浮点数。")
                try:
                    output_list.append(np.float32(item))
                except ValueError:
                    # 如果直接转换也失败，则抛出错误
                    raise ValueError(f"无法将 '{item}' 转换为浮点数或列表。请检查数据格式。")
        else:
            # 对于非字符串或不是列表字符串的元素，尝试直接转换为浮点数
            try:
                output_list.append(np.float32(item))
            except ValueError:
                raise ValueError(f"无法将 '{item}' 转换为浮点数。请检查数据格式。")

    # 尝试将处理后的列表堆叠成一个 NumPy 数组。
    # 这假设所有处理后的“元素”（例如，一个模式的展平向量）长度相同。
    # 如果长度不同，np.stack 会失败，这可能意味着您的数据结构需要填充或其他处理。
    try:
        print(f"[DEBUG] 每个清洗后元素 shape1: {[arr.shape for arr in output_list]}")
        return np.stack(output_list)
    except ValueError as e:
        print(f"警告：堆叠处理后的模式数组时失败，可能是由于长度不一致。错误：{e}")
        # 如果堆叠失败，回退到对象数组。但请注意，`euclidean_distances` 通常要求数值数组。
        print(f"[DEBUG] 每个清洗后元素 shape2: {[arr.shape for arr in output_list]}")
        return np.asarray(output_list, dtype=object)



# --- DataLoader 类定义 ---
class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        print(len(xs), batch_size)
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


# --- StandardScaler 类定义 ---
class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


# --- load_dataset 函数定义 ---
def load_dataset(dataset_dir, dataset_name, seq_len, traffic_state, batch_size, data_type):
    print(12)
    data = {}
    for category in ['train', 'eval', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, dataset_name, traffic_state, data_type, category + '.npz'),
                           allow_pickle=True)
        data['x_' + category] = cat_data['x']
        data['x_' + category] = data['x_' + category][:, 0:int(seq_len), :, :]
        data['y_' + category] = cat_data['y']
        data['y_' + category] = data['y_' + category][:, 0:int(seq_len), :, :]

    # print(123)
    # print(data.keys())
    # scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # # 数据格式转换
    # # 清洗 x 中潜在的字符串形式数组
    # for category in ['train', 'eval', 'test']:
    #     data['x_' + category] = _convert_string_lists_in_array(data['x_' + category])
    #     data['y_' + category] = _convert_string_lists_in_array(data['y_' + category])
    #
    # for category in ['train', 'eval', 'test']:
    #     data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    #     data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])
    #
    # data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=False)
    # data['eval_loader'] = DataLoader(data['x_eval'], data['y_eval'], batch_size, shuffle=False)
    # data['test_loader'] = DataLoader(data['x_test'], data['y_test'], batch_size, shuffle=False)
    # data['scaler'] = scaler
    print(123)

    scaler = StandardScaler(mean=data['x_train'][..., 1].mean(), std=data['x_train'][..., 1].std())
    # Data format
    for category in ['train', 'eval', 'test']:
        data['x_' + category][..., 1] = scaler.transform(data['x_' + category][..., 1])
        data['y_' + category][..., 1] = scaler.transform(data['y_' + category][..., 1])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=False)
    data['val_loader'] = DataLoader(data['x_eval'], data['y_eval'], batch_size, shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], batch_size, shuffle=False)
    data['scaler'] = scaler

    # --- 加载 pattern_keys 文件 ---
    final_pattern_keys = np.load(
        os.path.join(dataset_dir, dataset_name, f'pattern_keys_{dataset_name}.npy'),
        allow_pickle=True
    )

    # # --- 关键修改部分：加载和处理 pattern_keys ---
    # 1. 加载原始的 pattern_keys
    # 确保 allow_pickle=True，因为文件可能包含Python对象
    # raw_pattern_keys = np.load(
    #     os.path.join(dataset_dir, dataset_name, f'pattern_keys_{dataset_name}.npy'), allow_pickle=True
    # )
    #
    # # 2. 根据最新的 IndexError，我们知道 raw_pattern_keys[0] 是一个 NumPy 数组（而不是字典）
    # # 并且这个数组就是我们需要处理的含有字符串列表的模式数据。
    #
    # # 提取实际的数值模式数组 (raw_pattern_keys[0] 现在是您需要处理的 NumPy 数组)
    # x_train_raw_patterns_array = raw_pattern_keys[0]
    #
    # # 3. 对这个数组进行预处理
    # x_train_processed_patterns = _convert_string_lists_in_array(x_train_raw_patterns_array)
    #
    # # 4. 创建一个新的字典，以匹配 GATBlockWithPrompt 中 pattern_keys_tuple[0]['x_train'] 的期望结构
    # processed_pattern_dict = {'x_train': x_train_processed_patterns}
    #
    # # 5. 重构最终的 `pattern_keys` 元组
    # # 假设 raw_pattern_keys[1] 是其他不需要处理的辅助信息（如描述数组）。
    # # 如果 pattern_keys.npy 只有 raw_pattern_keys[0] 这一个元素，那么这里可能需要调整。
    # # 如果 raw_pattern_keys[1] 也需要清洗，也要在这里调用 _convert_string_lists_in_array。
    # final_pattern_keys = (processed_pattern_dict, raw_pattern_keys[1])
    # --- 关键修改部分结束 ---
    pattern_keys = np.load(os.path.join(dataset_dir, dataset_name,
                                        'pattern_keys_' + dataset_name  + '.npy'),allow_pickle=True)

    return data, final_pattern_keys