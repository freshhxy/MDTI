import os
import os.path as osp
import logging

from timm.scheduler import CosineLRScheduler

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import h5py

from tqdm import tqdm
from dataset.preprocess import Preprocess
from dataset.mdti_loader import TrajDataLoader
from model.MDTI import MDTI
from config.config import Config
import glob
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

# 在文件顶部添加辅助函数（与模型中保持一致）：
def get_unified_mask_positions(seq_len, mask_ratio=0.15, device='cpu'):
    """获取统一的mask位置，与模型中的逻辑保持一致"""
    num_mask = max(1, int(seq_len * mask_ratio))
    # 注意：这里需要使用相同的随机种子或者从模型传递mask_positions
    # 简化起见，我们使用固定的随机种子
    torch.manual_seed(1000)  # 固定种子确保一致性
    mask_positions = torch.randperm(seq_len, device=device)[:num_mask]
    mask_positions = mask_positions.sort()[0]
    torch.manual_seed(torch.initial_seed())  # 恢复随机种子
    return mask_positions
class Exp:
    def __init__(self, log_path):

        self.device = Config.device
        self.epochs = Config.training_epochs
        self.start_epoch = 0
        self.checkpoint_dir = log_path
        self.bad_patience = Config.bad_patience
        # 初始化预处理器（懒加载）
        self.preprocess = Preprocess()
        # 懒加载基础信息
        print("懒加载。。。。。")
        self.road_graph = self.preprocess.get_road_graph()
        self.grid_image = self.preprocess.get_grid_image()
        self.road_feature = self.preprocess.get_road_feature()
        # 更新配置
        Config.g_fea_size = self.road_feature.shape[1]
        Config.grid_num = self.preprocess.gs.grid_num
        Config.road_num = self.preprocess.edge.shape[0]
        Config.road_type = self.preprocess.edge.highway_type.nunique()
        # 初始化 dataloader 类，不立刻构造 loader
        self.dataloader = TrajDataLoader()
        self.train_loader = None
        self.eval_loader = None
        self.test_loader = None


        self.model = self._build_model()
        print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in self.model.parameters())))
        self.optimizer = self._build_optimize()
        self.lr_scheduler = self._build_schduler()



        self.writer = SummaryWriter(self.checkpoint_dir)

        self.train_mlm_loss_list = []
        self.train_cl_loss_list = []
        self.train_loss_list = []
        self.train_mlm_pred_list = []

        self.eval_loss_list = []
        self.eval_mlm_loss_list = []
        self.eval_cl_loss_list = []
        self.eval_mlm_pred_list = []

        self.best_eval_loss = 1e9

        self.model = self.model.to(self.device)
        self.best_save_config = None

        self.scaler = GradScaler()

    def _build_model(self):
        return MDTI()


    def _build_optimize(self):
        return optim.Adam(params=self.model.parameters(),
                          lr=Config.training_lr
                          )

    def _build_schduler(self):
        return CosineLRScheduler(optimizer=self.optimizer,
                                 t_initial=self.epochs,
                                 warmup_t=Config.warm_up_epoch,
                                 warmup_lr_init=Config.warmup_lr_init,
                                 lr_min=Config.lr_min
                                 )
    #
    # def save_model_with_epoch(self, epoch):
    #     save_config = dict()
    #     save_config['model'] = self.model.cpu()
    #     save_config['optimizer'] = self.optimizer.state_dict()
    #     save_config['lr_scheduler'] = self.lr_scheduler.state_dict()
    #     save_config['epoch'] = epoch
    #     save_config['best_eval_loss'] = self.best_eval_loss,
    #     save_config['train_loss_list'] = self.train_loss_list,
    #     save_config['train_cl_loss_list'] = self.train_cl_loss_list,
    #     save_config['train_mlm_loss_list'] = self.train_mlm_loss_list,
    #     save_config['eval_loss_list'] = self.eval_loss_list,
    #     save_config['eval_cl_loss_list'] = self.eval_cl_loss_list,
    #     save_config['eval_mlm_loss_list'] = self.eval_mlm_loss_list,
    #     save_config['train_mlm_pred_list'] = self.train_mlm_pred_list
    #     save_config['eval_mlm_pred_list'] = self.eval_mlm_pred_list
    #     cache_name = osp.join(self.checkpoint_dir, f'pretrain_model_{epoch}.pth')
    #     torch.save(save_config, cache_name)
    #     self.model.to(self.device)
    #     logging.info(f"Saved [{epoch} epoch] model at {cache_name}")
    #     return save_config
    def save_model_with_epoch(self, epoch):
        # 构造保存路径
        cache_name = osp.join(self.checkpoint_dir, f'pretrain_model_{epoch}.pth')

        # 构建更轻量的保存结构（只保存参数）
        save_config = {
            'model_state_dict': self.model.cpu().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'epoch': epoch,
            'best_eval_loss': self.best_eval_loss,
            'train_loss_list': self.train_loss_list,
            'train_cl_loss_list': self.train_cl_loss_list,
            'train_mlm_loss_list': self.train_mlm_loss_list,
            'eval_loss_list': self.eval_loss_list,
            'eval_cl_loss_list': self.eval_cl_loss_list,
            'eval_mlm_loss_list': self.eval_mlm_loss_list,
            'train_mlm_pred_list': self.train_mlm_pred_list,
            'eval_mlm_pred_list': self.eval_mlm_pred_list,
            'scaler_state_dict': self.scaler.state_dict()
        }

        # 保存模型
        torch.save(save_config, cache_name)
        self.model.to(self.device)
        logging.info(f"Saved [Epoch {epoch}] model at {cache_name}")

        # # 可选：清理旧模型，只保留最后1个
        # all_ckpts = sorted(glob.glob(osp.join(self.checkpoint_dir, "pretrain_model_*.pth")))
        # if len(all_ckpts) > 1:
        #     for ckpt in all_ckpts[:-1]:
        #         try:
        #             os.remove(ckpt)
        #             logging.info(f"Removed old checkpoint: {ckpt}")
        #         except Exception as e:
        #             logging.warning(f"Failed to remove {ckpt}: {e}")
        #
        return save_config

    def load_model_with_epoch(self, epoch):
        cache_name = osp.join(self.checkpoint_dir, f'pretrain_{epoch}.pth')
        assert os.path.exists(cache_name), f'Weights at {cache_name} not found'
        checkpoint = torch.load(cache_name, map_location='cpu')
        self.model = checkpoint['model'].to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_eval_loss = checkpoint['best_eval_loss']
        self.train_loss_list = list(checkpoint['train_loss_list'])
        self.train_mlm_loss_list = list(checkpoint['train_mlm_loss_list'])
        self.train_cl_loss_list = list(checkpoint['train_cl_loss_list'])
        self.eval_loss_list = list(checkpoint['eval_loss_list'])
        self.eval_mlm_loss_list = list(checkpoint['eval_mlm_loss_list'])
        self.eval_cl_loss_list = list(checkpoint['eval_cl_loss_list'])
        self.train_mlm_pred_list = list(checkpoint['train_mlm_pred_list'])
        self.eval_mlm_pred_list = list(checkpoint['eval_mlm_pred_list'])
        logging.info(f"Loaded model at {cache_name}")
        print(f"[AMP] autocast enabled: {torch.is_autocast_enabled()}")

    def train(self):

        bad_epoch = 0
        best_epoch = 0
        if self.train_loader is None:
            train_traj = self.preprocess.get_train_traj()
            self.train_loader = self.dataloader.get_data_loader(train_traj, is_shuffle=True,sample_ratio=0.17)
        if self.eval_loader is None:
            eval_traj = self.preprocess.get_eval_traj()
            self.eval_loader = self.dataloader.get_data_loader(eval_traj, is_shuffle=False,sample_ratio=0.17)

        print(f"训练集总样本数: {len(self.train_loader.dataset)}")
        print(f"每个 batch 大小: {self.train_loader.batch_size}")
        print(f"总 batch 数（iteration 次数）: {len(self.train_loader)}")
        for epoch in range(self.start_epoch, self.epochs):
            train_bar = tqdm(self.train_loader)
            train_loss = []
            train_mlm_loss = []
            train_cl_loss = []
            preds = []
            self.model.train()

            for i, batch_data in enumerate(train_bar):
                road_data, grid_data = batch_data
                for k, v in road_data.items():
                    if k == 'mask_road_index':
                        continue
                    road_data[k] = v.to(self.device)
                for k, v in grid_data.items():
                    grid_data[k] = v.to(self.device)
                # road_data['mask_road_index'] = (road_data['mask_road_index'][0].to(self.device), road_data['mask_road_index'][1].to(self.device))
                road_data['g_input_feature'] = torch.FloatTensor(self.road_feature).to(self.device)
                road_data['g_edge_index'] = torch.LongTensor(self.road_graph).to(self.device)
                grid_data['grid_image'] = torch.FloatTensor(self.grid_image).to(self.device)

                # cl_loss, mlm_loss, mlm_prediction = self.model(grid_data, road_data,self.pattern_keys)
                # # cl_loss, mlm_loss, mlm_prediction = self.model(grid_data, road_data)
                # loss = cl_loss + mlm_loss
                # self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()
                self.optimizer.zero_grad()

                with autocast():
                    cl_loss, mlm_loss, mlm_prediction= self.model(grid_data, road_data)

                # 用融合后的总损失
                loss = cl_loss+mlm_loss

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_loss.append(loss.item())
                train_cl_loss.append(cl_loss.item())
                train_mlm_loss.append(mlm_loss.item())
                # with torch.no_grad():
                #     mlm_label_prediction = mlm_prediction.argmax(dim=-1)  # (B*num_mask,)
                #     assert mlm_label_prediction.shape[0] == original_tokens.shape[0], \
                #         f"预测 {mlm_label_prediction.shape}, 标签 {original_tokens.shape}"
                #     correct_l = mlm_label_prediction.eq(original_tokens).float().mean().item()
                #     preds.append(correct_l)

                # with torch.no_grad():
                #     mlm_prediction = F.log_softmax(mlm_prediction, dim=-1)
                #     mlm_label = road_data['road_traj'][road_data['mask_road_index']]
                #     mlm_label_prediction = mlm_prediction.argmax(dim=-1)
                #     print("mlm_label shape:", mlm_label.shape)
                #     print("mlm_label_prediction shape:", mlm_label_prediction.shape)
                #
                #     correct_l = mlm_label.eq(mlm_label_prediction).sum().item() / mlm_label.shape[0]
                #     preds.append(correct_l)
                # with torch.no_grad():
                #     if mlm_prediction.numel() > 0:  # 只有非空才算
                #         mlm_prediction = F.log_softmax(mlm_prediction, dim=-1)
                #         mlm_label = road_data['road_traj'][road_data['mask_road_index']]
                #         mlm_label_prediction = mlm_prediction.argmax(dim=-1)
                #         correct_l = mlm_label.eq(mlm_label_prediction).sum().item() / mlm_label.shape[0]
                #         preds.append(correct_l)
                #     else:
                #         preds.append(0.0)  # 或者直接跳过
                with torch.no_grad():
                    mlm_prediction = F.log_softmax(mlm_prediction, dim=-1)

                    # 获取标签 - 与模型逻辑保持一致
                    road_traj = road_data['road_traj']
                    mask_positions = get_unified_mask_positions(road_traj.size(1), device=road_traj.device)
                    mlm_label = road_traj[:, mask_positions].reshape(-1)  # [batch_size * num_mask]

                    mlm_label_prediction = mlm_prediction.argmax(dim=-1)
                    correct_l = mlm_label.eq(mlm_label_prediction).sum().item() / mlm_label.shape[0]
                    preds.append(correct_l)
                train_bar.set_description(
                f'[MDTI Train Epoch {epoch}/{self.epochs}: total: {loss:.4f} | cl: {cl_loss.item():.4f} | mlm: {mlm_loss.item():.4f}]')

            with torch.no_grad():
                eval_loss, eval_cl_loss, eval_mlm_loss, eval_preds = self._eval(epoch)

                average_epoch_train_loss = np.array(train_loss).mean()
                average_epoch_train_cl_loss = np.array(train_cl_loss).mean()
                average_epoch_train_mlm_loss = np.array(train_mlm_loss).mean()

                self.train_loss_list.append(average_epoch_train_loss)
                self.train_cl_loss_list.append(average_epoch_train_cl_loss)
                self.train_mlm_loss_list.append(average_epoch_train_mlm_loss)

                average_epoch_eval_loss = np.array(eval_loss).mean()
                average_epoch_eval_cl_loss = np.array(eval_cl_loss).mean()
                average_epoch_eval_mlm_loss = np.array(eval_mlm_loss).mean()

                self.eval_loss_list.append(average_epoch_eval_loss)
                self.eval_cl_loss_list.append(average_epoch_eval_cl_loss)
                self.eval_mlm_loss_list.append(average_epoch_eval_mlm_loss)

                self.writer.add_scalar('loss/train_loss', average_epoch_train_loss, global_step=epoch)
                self.writer.add_scalar('loss/train_mlm_loss', average_epoch_train_mlm_loss, global_step=epoch)
                self.writer.add_scalar('loss/train_cl_loss', average_epoch_train_cl_loss, global_step=epoch)

                self.writer.add_scalar('loss/eval_loss', average_epoch_eval_loss, global_step=epoch)
                self.writer.add_scalar('loss/eval_mlm_loss', average_epoch_eval_mlm_loss, global_step=epoch)
                self.writer.add_scalar('loss/eval_cl_loss', average_epoch_eval_cl_loss, global_step=epoch)

                train_mlm_pred = np.array(preds).mean()
                eval_mlm_pred = np.array(eval_preds).mean()
                self.train_mlm_pred_list.append(train_mlm_pred)
                self.eval_mlm_pred_list.append(eval_mlm_pred)

                self.writer.flush()

                logging.info(
                    f'[Epoch {epoch}/{self.epochs}] [Train loss: {average_epoch_train_loss:.10f}] [Eval loss: {average_epoch_eval_loss:.10f}]'
                )
                logging.info(
                    f'[Epoch {epoch}/{self.epochs}] [Train cl  loss: {average_epoch_train_cl_loss:.10f}] [Eval cl  loss: {average_epoch_eval_cl_loss:.10f}]'
                )
                logging.info(
                    f'[Epoch {epoch}/{self.epochs}] [Train mlm loss: {average_epoch_train_mlm_loss:.10f}] [Eval mlm loss: {average_epoch_eval_mlm_loss:.10f}]'
                )
                logging.info(
                    f'[Epoch {epoch}/{self.epochs}] [Train mlm pred: {train_mlm_pred:.10f}] [Eval mlm pred: {eval_mlm_pred:.10f}]'
                )

                self.lr_scheduler.step(epoch + 1)
                if average_epoch_eval_loss >= self.best_eval_loss:
                    bad_epoch += 1

                if bad_epoch == self.bad_patience:
                    break

                tmp_config = self.save_model_with_epoch(epoch)
                if average_epoch_eval_loss < self.best_eval_loss:
                    bad_epoch = 0
                    self.best_eval_loss = average_epoch_eval_loss
                    self.best_save_config = tmp_config
                    best_epoch = epoch

        logging.info(f'Best model at [epoch {best_epoch}]!')
        cache_name = osp.join(self.checkpoint_dir, 'best_pretrain_model.pth')
        torch.save(self.best_save_config, cache_name)
        logging.info(f"Saved model at {cache_name}")

    def _eval(self, epoch):
        self.model.eval()
        eval_bar = tqdm(self.eval_loader)
        eval_loss = []
        eval_mlm_loss = []
        eval_cl_loss = []
        preds = []
        for i,batch_data in enumerate(eval_bar):
            road_data, grid_data = batch_data
            for k, v in road_data.items():
                if k == 'mask_road_index':
                    continue
                road_data[k] = v.to(self.device)
            for k, v in grid_data.items():
                grid_data[k] = v.to(self.device)
            # road_data['mask_road_index'] = (road_data['mask_road_index'][0].to(self.device), road_data['mask_road_index'][1].to(self.device))
            road_data['g_input_feature'] = torch.FloatTensor(self.road_feature).to(self.device)
            road_data['g_edge_index'] = torch.LongTensor(self.road_graph).to(self.device)
            grid_data['grid_image'] = torch.FloatTensor(self.grid_image).to(self.device)
            #

            # with h5py.File(prompt_emb_path, 'r') as f:
            #     prompt_embedding = torch.tensor(f['embeddings'][:]).to(self.device)

            # cl_loss, mlm_loss, mlm_prediction = self.model(grid_data, road_data, prompt_embedding)
            cl_loss, mlm_loss, mlm_prediction = self.model(grid_data, road_data)
            # cl_loss, mlm_loss, mlm_prediction, original_tokens = self.model(grid_data, road_data)

            loss = cl_loss+mlm_loss

            eval_loss.append(loss.item())
            eval_cl_loss.append(cl_loss.item())
            eval_mlm_loss.append(mlm_loss.item())

            eval_bar.set_description(
                f'[MDTI Eval  Epoch {epoch}/{self.epochs}: total: {loss:.4f} | cl: {cl_loss.item():.4f} | mlm: {mlm_loss.item():.4f}]')
            # with torch.no_grad():
            #     mlm_label_prediction = mlm_prediction.argmax(dim=-1)
            #     assert mlm_label_prediction.shape[0] == original_tokens.shape[0], \
            #         f"预测 {mlm_label_prediction.shape}, 标签 {original_tokens.shape}"
            #     correct_l = mlm_label_prediction.eq(original_tokens).float().mean().item()
            #     preds.append(correct_l)

            mlm_prediction = F.log_softmax(mlm_prediction, dim=-1)
            road_traj = road_data['road_traj']
            mask_positions = get_unified_mask_positions(road_traj.size(1), device=road_traj.device)
            mlm_label = road_traj[:, mask_positions].reshape(-1)  # [batch_size * num_mask]

            mlm_label_prediction = mlm_prediction.argmax(dim=-1)
            correct_l = mlm_label.eq(mlm_label_prediction).sum().item() / mlm_label.shape[0]
            preds.append(correct_l)
            # mlm_prediction = F.log_softmax(mlm_prediction, dim=-1)
            # mlm_label = road_data['road_traj'][road_data['mask_road_index']]
            # mlm_label_prediction = mlm_prediction.argmax(dim=-1)
            # correct_l = mlm_label.eq(mlm_label_prediction).sum().item() / mlm_label.shape[0]
            #
            # preds.append(correct_l)

        return eval_loss, eval_cl_loss, eval_mlm_loss, preds
