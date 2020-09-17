__all__ = ('Dice', 'BCE', 'BCEDice',)

import torch
import torch.nn.functional as F


class Dice:
    """Dice loss & metric."""

    def __init__(self, p=2, smooth=1., reduction='mean'):
        self.p = p
        self.smooth = smooth
        self.reduction = reduction
        print("Train/Eval with Dice Loss/Metric")

    def __call__(self, pred, label):
        """
            Args:
                pred: output of network, with shape Nx1xHxW;
                label: label of image, with shape NxHxW;
            Return:
                avg multi label dice loss overage all pixels.
        """

        prob = torch.sigmoid(pred.squeeze(1))

        N = label.shape[0]
        # (N,H,W) -> (N,H*W)
        m1 = torch.reshape(prob, (N, -1))
        m2 = torch.reshape(label, (N, -1))

        # (N,)
        intersection = (m1 * m2).sum(dim=1)
        # 使用 p > 1 可使loss变大，加速收敛
        union = m1.pow(self.p).sum(dim=1) + m2.pow(self.p).sum(dim=1)
        coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        # loss over batch data
        loss = 1. - coeff

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            raise NotImplementedError("Not implemented with reduction '{}'".format(self.reduction))

        return loss
        # # 评估模式则同时返回dice评估系数
        # if eval:
        #     cost = self._dice(pred, label)
        #     return loss.detach().item(), cost.detach().item()
        # else:
        #     return loss

    def get_dice(self, pred, label):
        # (N,1,H,W)->(N,H,W)
        pred = torch.sigmoid(pred.squeeze(1))
        pred[pred >= .5] = 1
        pred[pred < .5] = 0

        batch = label.shape[0]
        # (N,H,W)->(N,H*W)
        label = label.view(batch, -1)
        pred = pred.view(batch, -1)

        inter = 2 * (pred * label).sum(dim=1)
        union = pred.sum(dim=1) + label.sum(dim=1)
        dice = inter / union

        if self.reduction == 'mean':
            return dice.mean().detach().item()
        elif self.reduction == 'sum':
            return dice.sum().detach().item()
        elif not self.reduction:
            return dice.detach()
        else:
            raise NotImplementedError("Not implemented with reduction '{}'".format(self.reduction))


class BCE(torch.nn.BCEWithLogitsLoss):
    """self-defined BCEWithLogitsLoss, details see torch.nn.BCEWithLogitsLoss."""

    def __init__(self, weight=None, reduction='mean', pos_weight=None):
        super(BCE, self).__init__(weight=weight, reduction=reduction, pos_weight=pos_weight)
        print("#Train with BCEWithLogitsLoss\n")

    def forward(self, pred, target):
        # pred - (N,1,H,W); target - (N,H,W)
        # BCE要求预测和标签是float类型
        return super(BCE, self).forward(pred.squeeze(1), target.float())


class BCEDice:
    """Combine BCEWithLogitsLoss with DiceLoss.
       pos_weight是正样本的bce loss权重，通常设置为正负样本比的倒数。"""

    def __init__(self, pos_weight=None, p=2, smooth=1., reduction='mean'):
        self.bce = BCE(pos_weight=pos_weight)  # torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = Dice(p, smooth, reduction)
        print("#Train with BCEWithLogitsLoss & DiceLoss\n")

    def __call__(self, pred, target):
        # pred - (N,1,H,W); target - (N,H,W)
        # BCELoss要求预测和标签的shape一致
        bce_loss = self.bce(pred, target)  # self.bce(pred.squeeze(1), target.float())
        dice_loss = self.dice(pred, target)
        loss = 2. * bce_loss + dice_loss

        return loss, bce_loss.detach().item(), dice_loss.detach().item()
