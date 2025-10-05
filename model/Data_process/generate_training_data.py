
import argparse
import os

import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans, KShape
from data_process_2 import StandardScaler
def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=True, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)  # shape: (T, N, 1)

    feature_list = [data]  # 只保留数值特征

    if add_time_in_day:
        total_seconds = 24 * 3600
        time_ind = (
            df.index.hour * 3600 + df.index.minute * 60 + df.index.second
        ) / total_seconds  # 归一化到 [0, 1]
        time_in_day = np.tile(time_ind[:, None, None], [1, num_nodes, 1]).transpose((2, 1, 0))
        # 最终 shape: (T, N, 1)
        feature_list.append(time_in_day)

    if add_day_in_week:
        # dow = df.index.dayofweek  # 0=Monday, ..., 6=Sunday
        # dow_onehot = np.eye(7)[dow]  # shape: (T, 7)
        # dow_onehot = np.tile(dow_onehot[:, None, :], [1, num_nodes, 1])  # (T, N, 7)
        # feature_list.append(dow_onehot)
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    data = np.concatenate(feature_list, axis=-1)  # shape: (T, N, input_dim)

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    return x, y



def generate_train_val_test(args):
    cluster_method = "kshape"
    points_per_hour = 3600 // args.time_intervals

    points_per_day = 24 * 3600 // args.time_intervals

    s_attn_size = 3

    cand_key_days = 14

    n_cluster = 16

    cluster_max_iter = 5
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    #METR_LA
    # df = pd.read_hdf(args.traffic_df_filename)
    #PEMS_lane
    df = pd.read_csv(args.traffic_df_filename,index_col= 0)
    # 只保留数值型列（过滤掉字符串/对象型列）
    df = df.select_dtypes(include=[np.number])
    # print(pd.to_datetime(df.index.values).astype(str))
    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=False,
        add_day_in_week=args.dow,
    )

    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    print("x shape: ", x.shape, ", y shape: ", y.shape)

    # Write the data into npz file.



    x_train, y_train = x[:num_train], y[:num_train]
    x_eval, y_eval = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]


    # scaler = StandardScaler(mean=x_train[..., 1].mean(), std=x_train[..., 1].std())
    scaler = StandardScaler(
        mean=x_train[..., 0].astype(np.float32).mean(),
        std=x_train[..., 0].astype(np.float32).std()
    )

    x_train_scaler = np.expand_dims(scaler.transform(x_train[..., 0]),axis=-1)

    pattern_key_file = os.path.join(args.data_dir, 'pattern_keys_{}'.format(
        args.data_name))

    print("处理 pattern_keys")

    print(os.path.exists(pattern_key_file + '.npy'))

    if not os.path.exists(pattern_key_file + '.npy'):
        cand_key_time_steps = cand_key_days * points_per_day
        print(x_train_scaler.shape)
        pattern_cand_keys = (x_train_scaler[:cand_key_time_steps, :s_attn_size, :, :1]
                             .swapaxes(1, 2)
                             .reshape(-1, s_attn_size, 1))

        if cluster_method == "kshape":
            km = KShape(n_clusters=n_cluster, max_iter=cluster_max_iter).fit(pattern_cand_keys)
        else:
            km = TimeSeriesKMeans(n_clusters=n_cluster, metric="softdtw", max_iter=cluster_max_iter).fit(
                pattern_cand_keys)

        pattern_keys = km.cluster_centers_
        pattern_dict_to_save = {
            'x_train': km.cluster_centers_  # 这里的 km.cluster_centers_ 形状是 (16, 3, 1)
            # 如果有其他 pattern keys，也在这里添加
        }
        # 假设 pattern_keys 只是一个包含这个字典的元组
        pattern_keys_final = (pattern_dict_to_save,)  # 确保它是一个 tuple

        np.save(pattern_key_file, pattern_keys_final, allow_pickle=True)  # 加上 allow_pickle=True
        # np.save(pattern_key_file, pattern_keys)
        print("Saved at file " + pattern_key_file + ".npy")
    else:
        pattern_keys = np.load(pattern_key_file + ".npy")
        print("Loaded file " + pattern_key_file + ".npy")


    for cat in ["train", "eval", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz" ),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./data/xian/rn/traj", help="Output directory.")
    parser.add_argument("--traffic_df_filename", type=str, default="./data/xian/traj.csv", help="Raw traffic readings.")
    parser.add_argument("--time_intervals", type=int, default="300",
                        help="if 1min is 60 ,if 5min is 300" )
    parser.add_argument("--data_dir", type=str, default="./data/xian", help="patten_key directory.only road")

    parser.add_argument("--data_name", type=str, default="xian", help="patten_key directory.")
    parser.add_argument("--seq_length_x", type=int, default=12, help="Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=12, help="Sequence Length.",)
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start", )

    parser.add_argument("--dow",default=0, action='store_true',)

    args = parser.parse_args()
    # if os.path.exists(args.output_dir):
    #     reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
    #     if reply[0] != 'y': exit
    # else:
    #     os.makedirs(args.output_dir)
    generate_train_val_test(args)
