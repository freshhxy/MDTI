import numpy as np
import os
class DataLoader (object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        print (len (xs), batch_size)
        if pad_with_last_sample:
            num_padding = (batch_size - (len (xs) % batch_size)) % batch_size
            x_padding = np.repeat (xs[-1:], num_padding, axis=0)
            y_padding = np.repeat (ys[-1:], num_padding, axis=0)
            xs = np.concatenate ([xs, x_padding], axis=0)
            ys = np.concatenate ([ys, y_padding], axis=0)
        self.size = len (xs)
        self.num_batch = int (self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation (self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min (self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper ()


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


def load_dataset(dataset_dir,dataset_name,seq_len, batch_size,):
    print(12)
    data = {}
    for category in ['train', 'eval', 'test']:
        file_path = os.path.join(dataset_dir, dataset_name, category + '.npz')
        print(f"[DEBUG] 正在加载文件: {file_path}")
        cat_data = np.load(file_path, allow_pickle=True)
        print(f"[DEBUG] 包含 keys: {cat_data.files}")
        print(f"[DEBUG] x shape: {cat_data['x'].shape}, y shape: {cat_data['y'].shape}")

        data['x_' + category] = cat_data['x']
        data['x_' + category] = data['x_' + category][:, 0:int (seq_len), :, :]
        data['y_' + category] = cat_data['y']
        data['y_' + category] = data['y_' + category][:,0:int(seq_len),:,:]

    print(123)

    scaler = StandardScaler(mean=data['x_train'][..., 1].mean(), std=data['x_train'][..., 1].std())
    # Data format
    for category in ['train', 'eval', 'test']:
        data['x_' + category][..., 1] = scaler.transform(data['x_' + category][..., 1])
        data['y_' + category][..., 1] = scaler.transform(data['y_' + category][..., 1])
    data['train_loader'] = DataLoader(data['x_train'] , data['y_train'] , batch_size, shuffle=False)
    data['eval_loader'] = DataLoader(data['x_eval'] , data['y_eval'], batch_size, shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'] , data['y_test'] , batch_size, shuffle=False)
    data['scaler'] = scaler

    pattern_keys = np.load(os.path.join(dataset_dir, dataset_name,  'pattern_keys_'+dataset_name+ '.npy'),allow_pickle=True)

    return data , pattern_keys

