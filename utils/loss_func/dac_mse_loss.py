import torch
import torch.nn as nn
import torch.nn.functional as F

class DAC_MSE_loss(nn.Module):
    """
    this loss function should support mse loss and infoNCE loss.
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device

    def mse_loss(self, pred, target):
        """
        Args:
            pred (Tensor): NxC input features.
            target (Tensor): NxC target features.
        """
        N = pred.size(0)
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        loss = 1 - 1 * (pred_norm * target_norm).sum() / N
        return loss

    def forward(self, image_features1, image_features2):
        b, c, n = image_features1.shape
        # feat1 = image_features1.transpose(2, 1).reshape(b * n, c)  #  这里对比原来的方法少了一个mlp映射，相当于少了一个特征空间对齐，交给backbone去做吧
        # feat2 = image_features2.transpose(2, 1).reshape(b * n, c)

        feat1 = image_features1.transpose(2, 1).reshape(b, c*n)  # 这里对比原来的方法少了一个mlp映射，相当于少了一个特征空间对齐，交给backbone去做吧
        feat2 = image_features2.transpose(2, 1).reshape(b, c*n)

        loss = self.mse_loss(feat1, feat2)
        return loss

class KD_MSE_loss(nn.Module):
    """
    this loss function should support mse loss and infoNCE loss.
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device

    def mse_loss(self, pred, target):
        """
        Args:
            pred (Tensor): NxC input features.
            target (Tensor): NxC target features.
        """
        N = pred.size(0)
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        loss = 1 - 1 * (pred_norm * target_norm).sum() / N
        return loss

    def forward(self, image_features1, image_features2):
        b, c, n = image_features1.shape

        feat1 = image_features1.transpose(2, 1).reshape(b, c * n)  # 这里对比原来的方法少了一个mlp映射，相当于少了一个特征空间对齐，交给backbone去做吧
        feat2 = image_features2.transpose(2, 1).reshape(b, c * n)

        loss = self.mse_loss(feat1, feat2)
        return loss