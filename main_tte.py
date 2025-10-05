import os
import logging
import argparse
import warnings

from config.config import Config
from Exp.tte_trainer import Exp
# from model.Data_process.data_process_2 import load_dataset # 导入加载数据集的函数


def parse_args():
    # dont set default value here! -- it will incorrectly overwrite the values in config.py.
    # config.py is the correct place for default values.
    parser = argparse.ArgumentParser(description='Train Travel Time Estimation')
    parser.add_argument('--dumpfile_uniqueid', type=str, help='see config.py')
    parser.add_argument('--dataset', type=str, help='')

    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))


def get_log_path(exp_id):
    log_path = f'./log/{Config.dataset}/{exp_id}/tte'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    return log_path


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    Config.update(parse_args())

    exp_id =  f'exp_bs128_road{Config.road_trm_layer}_grid{Config.grid_trm_layer}_inter{Config.inter_trm_layer}_epoch30_2e-4_cls_{Config.mask_length}_{Config.mask_ratio}ratio_10'

    log_path = get_log_path(exp_id)
    pretrain_path = f'./log/{Config.dataset}/{exp_id}/pretrain/pretrain_model_29.pth'

    logging.basicConfig(level=logging.INFO,
                        format="[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
                        handlers=[logging.FileHandler(log_path + '/train_log.log', mode='w'),
                                  logging.StreamHandler()]
                        )
    # Config.training_lr = 1e-4
    print("Args in experiment:")
    logging.info('=================================')
    logging.info(Config.to_str())
    logging.info('=================================')

    # pattern_keys = load_dataset(
    #     dataset_dir=Config.dataset_dir,
    #     dataset_name=Config.dataset,
    #     seq_len=Config.seq_len,
    #     traffic_state=Config.traffic_state,
    #     batch_size=Config.batch_size,
    #     data_type=Config.data_type,
    # )
    # exp = Exp(log_path, pretrain_path)
    # # ---- 这里加断点恢复 ----
    #
    # #
    # ckpt = os.path.join(log_path, 'tte_best.pth')
    # if os.path.exists(ckpt):
    #     exp.load_model_with_epoch()

    exp = Exp(log_path, pretrain_path)
    exp.train()

