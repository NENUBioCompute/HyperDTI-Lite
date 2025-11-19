import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# 单卡loss
class CustomLoss(nn.Module):
    def __init__(self, manifold, dim, anchor_factor, loss_rate):
        super(CustomLoss, self).__init__()
        self.manifold = manifold
        self.dim = dim
        self.anchor_factor = anchor_factor
        self.loss_rate = loss_rate

    def forward(self, prediction, labels):
        anchor = self.manifold.expmap0(self.anchor_factor * torch.ones(self.dim).cuda())
        score = -self.manifold.dist(prediction, anchor)
        loss_pos = labels * F.softplus(-score)  # 正数部分，≈x
        loss_neg = self.loss_rate * (1 - labels) * F.softplus(score)  # 负数部分，接近于0
        loss = loss_pos + loss_neg
        return loss.sum()

# #多卡loss
# class CustomLoss(nn.Module):
#     def __init__(self, manifold, dim, anchor_factor, loss_rate):
#         super(CustomLoss, self).__init__()
#         self.manifold = manifold
#         self.dim = dim
#         self.anchor_factor = anchor_factor
#         self.loss_rate = loss_rate
#
#     def forward(self, prediction, labels):
#         if labels.dim() > 1:
#             labels = labels.view(-1)
#
#         batch_size = prediction.size(0)
#         anchor_vec = torch.ones(self.dim, device=prediction.device) * self.anchor_factor
#         anchor = self.manifold.expmap0(anchor_vec).expand(batch_size, -1)
#
#         score = -self.manifold.dist(prediction, anchor)
#
#         if labels.shape[0] != score.shape[0]:
#             # 修复多 GPU 下 DataParallel 导致的 batch 扩展
#             labels = labels.repeat_interleave(score.shape[0] // labels.shape[0])
#
#         loss_pos = labels * F.softplus(-score)
#         loss_neg = self.loss_rate * (1 - labels) * F.softplus(score)
#         loss = loss_pos + loss_neg
#         return loss.sum()

""" early stop """
class EarlyStopping:
    def __init__(self, path=None, patience=7, verbose=False, delta=0, fold=None):
        """
        Args:
            patience (int): 连续多少个epoch验证集损失不减小时，停止训练
            verbose (bool): 如果为True，每次验证集损失减少时会输出消息
            delta (float): 验证集损失减小的阈值
            path (str): 模型保存路径
            fold (int): 标志模型折次，用于区分保存的模型路径
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.fold = fold

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """"保存验证集损失减小的模型"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path+ f'/fold_{self.fold}_valid_best_checkpoint.pth')
        self.val_loss_min = val_loss