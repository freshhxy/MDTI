import os
import logging
import argparse
import warnings

from config.config import Config
from Exp.mdti_trainer import Exp
# from model.Data_process.data_process import load_dataset # 导入加载数据集的函数

import psutil
import time

def parse_args():
    # dont set default value here! -- it will incorrectly overwrite the values in config.py.
    # config.py is the correct place for default values.
    parser = argparse.ArgumentParser(description='Train MDTI')
    parser.add_argument('--dumpfile_uniqueid', type=str, help='see config.py')
    parser.add_argument('--dataset', type=str, help='')

    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))


def get_log_path(exp_id):
    log_name = f'./log/{Config.dataset}/{exp_id}/pretrain'
    if not os.path.exists(log_name):
        os.makedirs(log_name)
    return log_name

def wait_for_memory(threshold_gb=120, check_interval=10):
    while True:
        mem = psutil.virtual_memory()
        used_gb = (mem.total - mem.available) / (1024 ** 3)
        print(f"[Memory Check] Used: {used_gb:.2f} GB / Threshold: {threshold_gb} GB")
        if used_gb < threshold_gb:
            print("Memory usage is acceptable. Proceeding...")
            break
        else:
            print("Memory too high. Waiting...")
            time.sleep(check_interval)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    Config.update(parse_args())

    exp_id = f'exp_bs128_road{Config.road_trm_layer}_grid{Config.grid_trm_layer}_inter{Config.inter_trm_layer}_epoch30_2e-4_cls_{Config.mask_length}_{Config.mask_ratio}ratio'
    log_path = get_log_path(exp_id)

    logging.basicConfig(level=logging.INFO,
                        format="[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
                        handlers=[logging.FileHandler(log_path + '/train_log.log', mode='w'),
                                  logging.StreamHandler()]
                        )


    print("Args in experiment:")
    logging.info('=================================')
    logging.info(Config.to_str())
    logging.info('=================================')

    try:
        # 用法示例
        print("33")
        # wait_for_memory(threshold_gb=120)
        # exp = Exp(log_path,pattern_keys)
        exp = Exp(log_path)

        exp.train()
    except Exception as e:
        logging.exception("Exception during training:")
    finally:
        import os, signal

        logging.info("Terminating process cleanly.")

