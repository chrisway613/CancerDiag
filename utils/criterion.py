__all__ = ('Dice',)

import torch
import torch.nn.functional as F


class Dice:
    """Dice loss & metric."""

    def __init__(self, p=2, smooth=1., reduction='mean'):
        self.p = p
        self.smooth = smooth
        self.reduction = reduction

    def __call__(self, pred, label, eval=False):
        """
            Args:
                pred: output of network, with shape Nx1xHxW;
                label: label of image, with shape NxHxW;
                eval: whether to infer on evaluation set
            Return:
                avg multi label dice loss overage all pixels.
        """

        prob = F.sigmoid(pred.squeeze(1))

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

        # 评估模式则同时返回dice评估系数
        if eval:
            cost = self._dice(pred, label)
            return loss.detach().item(), cost.detach().item()
        else:
            return loss

    def _dice(self, pred, label):
        # (N,1,H,W)->(N,H,W)
        pred = F.sigmoid(pred.squeeze(1))
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
            return dice.mean()
        elif self.reduction == 'sum':
            return dice.sum()
        elif not self.reduction:
            return dice
        else:
            raise NotImplementedError("Not implemented with reduction '{}'".format(self.reduction))
