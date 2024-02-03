import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.stats as stats


class My_loss(nn.Module):
    def __init__(self, alpha=0.01):
        super(My_loss, self).__init__()
        self.alpha = alpha

    def forward(self, source, target):
        source_np = source.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        kendall_coeff, _ = stats.kendalltau(source_np, target_np)
        kendall_loss = -torch.tensor(kendall_coeff, dtype=torch.float32, requires_grad=True)

        mse_loss = F.mse_loss(source, target)
        loss = self.alpha * kendall_loss + mse_loss
        return loss, kendall_loss, mse_loss
        