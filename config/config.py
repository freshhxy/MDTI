import random
import numpy
import torch


# def set_seed(seed=-1):
#     if seed == -1:
#         return
#     random.seed(seed)
#     numpy.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


class Config:
    is_resume = False
    is_cls = True
    # seed = 1000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cuda:9")

    # dataset
    dataset = 'porto'
    dataset_dir = './data'
    traffic_state='rn'
    data_type = "traj"
    seq_len = 12
    mode = 'Road'
    grid_special_tokens = {
        'padding_token': 0,
        'cls_token': 1,
    }

    road_special_tokens = {
        'padding_token': 0,
        'cls_token': 1,
        'mask_token': 2,
    }

    mask_length = 2
    mask_ratio = 0.2

    grid_size = 100
    grid_num = 0

    road_num = 0
    #porto
    road_type = 8
    #chengdu
    # road_type = 13
    # #xian
    # road_type =  11
    batch_size = 32
    training_epochs = 30
    training_lr = 2e-4
    bad_patience = 5

    # cos scheduler
    warm_up_epoch = 10
    warmup_lr_init = 1e-6
    lr_min = 1e-6
    weight_decay = 0.02

    # ===========Common Setting=========
    hidden_emb_dim = 256
    pe_dropout = 0.1
    out_emb_dim = 256

    # ===========Grid Encoder==============
    grid_in_channel = 3
    in_channel = 64
    grid_out_channel = 64
    # grid_out_channel = 256
    grid_trm_head = 4
    grid_trm_dropout = 0.1
    grid_trm_layer = 2
    grid_ffn_dim = hidden_emb_dim * 4

    # Grid Image Dimensions (需要根据实际图像尺寸设置)
    # porto
    grid_H = 114  # 例如，您的图像高度
    grid_W = 52  # 例如，您的图像宽度
    # chengdu
    # grid_H = 264  # 例如，您的图像高度
    # grid_W = 246  # 例如，您的图像宽度
    # # xian
    # grid_H = 111 # 例如，您的图像高度
    # grid_W = 112 # 例如，您的图像宽度
    # GAT Configuration (新加入)
    gat_hidden_channels = 128  # GAT 隐藏层的通道数
    # gat_num_heads = 4  # GAT 每个 GATConv 层的注意力头数量
    gat_num_heads = 3
    gat_num_layers = 3  # GAT 的层数 (至少 2 层：输入层 -> 隐藏层... -> 输出层)
    gat_dropout = 0.6  # GAT 的 dropout 率

    w_cl = 1.0
    w_mlm = 0.7
    w_eta = 1.0
    w_step = 0.3
    w_mono = 0.1
    eta_target_scale = 60  # 分钟尺度建模时长；若你的模型里按分钟对数正态，推理时会 *scale 回秒*

    # ===========Road Encoder=============
    g_fea_size = 0
    g_heads_per_layer = [4, 4, 4]
    g_dim_per_layer = [hidden_emb_dim, hidden_emb_dim, hidden_emb_dim]
    g_num_layers = 3
    g_dropout = 0.1

    road_trm_head = 4
    road_trm_dropout = 0.1
    road_trm_layer = 4
    road_ffn_dim = hidden_emb_dim * 4

    # ===========Interactor=============
    inter_trm_head = 2
    inter_trm_dropout = 0.1
    inter_trm_layer = 2
    inter_ffn_dim = out_emb_dim * 4

    lambda_cross_modal_relation = 0.5  # 加权系数
    cm_w_step = 1.0
    cm_w_rel = 1.0
    cm_w_cos = 1.0

    # downstream tasks
    tuning_all = True
    prediction_length = 5

    @classmethod
    def update(cls, dic: dict):
        for k, v in dic.items():
            if k in cls.__dict__:
                assert type(getattr(Config, k)) == type(v)
            setattr(Config, k, v)
        cls.post_value_updates()

    @classmethod
    def post_value_updates(cls):
        if 'porto' == cls.dataset:
            cls.min_lon = -8.68942
            cls.min_lat = 41.13934
            cls.max_lon = -8.55434
            cls.max_lat = 41.18586
            cls.cls_num = 3
            cls.weight_decay = 0.0001

        elif 'rome' == cls.dataset:
            cls.min_lon = 12.37351
            cls.max_lon = 12.61587
            cls.min_lat = 41.79417
            cls.max_lat = 41.99106
            cls.cls_num = 315
        elif 'chengdu' == cls.dataset:
            cls.min_lon = 103.92994
            cls.max_lon = 104.20535
            cls.min_lat = 30.56799
            cls.max_lat = 30.78813
            cls.cls_num = 2
            cls.weight_decay = 0.0001
        elif 'xian' == cls.dataset:
            cls.min_lon = 108.92099
            cls.max_lon = 109.00970
            cls.min_lat = 34.20421
            cls.max_lat = 34.27934
            cls.cls_num = 5  # 按数据集情况改
            cls.weight_decay = 1e-4  # 与其他城市保持一致

        else:
            raise NotImplementedError

        # set_seed(cls.seed)

    @classmethod
    def to_str(cls):
        dic = cls.__dict__.copy()
        lst = list(filter(
            lambda p: (not p[0].startswith('__')) and type(p[1]) != classmethod,
            dic.items()
        ))
        return '\n'.join([str(k) + ' = ' + str(v) for k, v in lst])
