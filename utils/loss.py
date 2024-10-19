import os, argparse, time
import numpy as np
import time
import torch
import torch.nn as nn
from torch.nn import functional as F


class L2RankLoss(torch.nn.Module):
    """
    L2 loss (mos) + Rank loss (regression mos) + cross-entropy loss(classification)
    (except the rank loss, we can also try triplet loss for regression)
    """

    def __init__(self, args, **kwargs):
        super(L2RankLoss, self).__init__()
        self.l2_w = 1
        self.rank_w = 1
        self.hard_thred = 1
        self.use_margin = False
        self.args = args

    def forward(self, preds_reg, preds_cls, gts, dt):
        # 1. Regression loss
        preds_reg = preds_reg.view(-1)
        gts = gts.view(-1)
        # l1 loss
        l2_loss = F.mse_loss(preds_reg, gts) * self.l2_w

        # simple rank
        n = len(preds_reg)
        preds_reg = preds_reg.unsqueeze(0).repeat(n, 1)
        preds_reg_t = preds_reg.t()
        img_label = gts.unsqueeze(0).repeat(n, 1)
        img_label_t = img_label.t()
        masks = torch.sign(img_label - img_label_t)
        masks_hard = (torch.abs(img_label - img_label_t) < self.hard_thred) & (torch.abs(img_label - img_label_t) > 0)
        if self.use_margin:
            rank_loss = masks_hard * torch.relu(torch.abs(img_label - img_label_t) - masks * (preds_reg - preds_reg_t))
        else:
            rank_loss = masks_hard * torch.relu(- masks * (preds_reg - preds_reg_t))
        rank_loss = rank_loss.sum() / (masks_hard.sum() + 1e-08)
        loss_regression = l2_loss + rank_loss * self.rank_w

        # 2. Classification loss
        if self.args.use_classificaiton:
            loss_classification = F.cross_entropy(preds_cls, dt.squeeze(1))
        else:
            loss_classification = 0

        # Total loss
        alpha = 0.5
        loss_total = alpha*loss_regression + (1-alpha)*loss_classification
        return loss_total, loss_regression, loss_classification

