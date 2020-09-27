__all__ = ('Dice', 'BCE', 'BCEDice',)

import torch


class Dice:
    """Dice loss(with OHEM, optional) & metric."""

    def __init__(self, p=2, smooth=1e-8, reduction='mean', ohem=False, top_k_ratio=1.):
        if ohem:
            assert 0 < top_k_ratio < 1, "'top_k_ratio' must in range (0, 1) when in OHEM mode"
        self.ohem = ohem
        self.top_k_ratio = top_k_ratio

        self.p = p
        self.smooth = smooth
        self.reduction = reduction

        print("#Train/Eval with Dice Loss/Metric\n")

    def __call__(self, pred, label):
        """
            Args:
                pred: output of network, with shape Nx1xHxW;
                label: label of image, with shape NxHxW;
            Return:
                avg multi label dice loss overage all pixels.
        """

        N = label.shape[0]
        # (N,H*W)
        label = label.view(N, -1)
        pred = pred.view(N, -1)
        # prob = torch.sigmoid(pred)
        # m1 = label
        # m2 = prob
        #
        # # (N,)
        # intersection = (m1 * m2).sum(dim=1)
        # # 使用 p > 1 可使loss变大，加速收敛
        # union = m1.pow(self.p).sum(dim=1) + m2.pow(self.p).sum(dim=1)
        # coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        # # (N,)
        # loss = 1. - coeff
        # assert loss.shape[0] == N and loss.ndim == 1

        # 只对阳性样本计算dice loss
        pos_indices = torch.where(label.sum(dim=1) > 0)[0]
        if len(pos_indices):
            N = len(pos_indices)
            # (num_pos,)
            m1 = label[pos_indices]
            m2 = torch.sigmoid(pred[pos_indices])

            # (num_pos,)
            intersection = (m1 * m2).sum(dim=1)
            # 使用 p > 1 可使loss变大，加速收敛
            union = m1.pow(self.p).sum(dim=1) + m2.pow(self.p).sum(dim=1)
            coeff = (2. * intersection + self.smooth) / (union + self.smooth)
            # (num_pos,) loss over positive samples
            loss = 1. - coeff
            assert loss.shape == pos_indices.shape and loss.ndim == 1
        else:
            # 若该batch全是阴性样本，则dice loss置0
            loss = torch.zeros(N, requires_grad=True, device=label.device)

        if self.ohem:
            # OHEM模式下，只取loss最大的topk样本进行学习
            keep_num = max(1, int(self.top_k_ratio * N))
            loss, indices = loss.topk(keep_num)
            if len(pos_indices):
                # 阳性样本的索引映射到原批次索引
                indices = pos_indices[indices]
        else:
            # 没有阳性样本时索引就是原批次索引
            indices = pos_indices if len(pos_indices) else torch.arange(N)

        if self.reduction == 'mean':
            return torch.mean(loss), indices
        elif self.reduction == 'sum':
            return torch.sum(loss), indices
        elif self.reduction == 'none':
            return loss, indices
        else:
            raise NotImplementedError("Not implemented with reduction '{}'".format(self.reduction))

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
    """torch.nn.BCEWithLogitsLoss with OHEM."""

    def __init__(self, weight=None, reduction='mean', pos_weight=None, ohem=False, top_k_ratio=1.):
        if ohem:
            assert 0 < top_k_ratio < 1, "'top_k_ratio' must in range (0, 1) when in OHEM mode"
            super(BCE, self).__init__(weight=weight, reduction='none', pos_weight=pos_weight)
        else:
            super(BCE, self).__init__(weight=weight, reduction=reduction, pos_weight=pos_weight)

        # 用另一个属性来记录reduction,因为OHEM时要用reduction='none'来实例化pytorch的BCE，
        # 这样BCE返回的loss才能是每个样本的loss，以便下一步筛选topk
        self._reduction = reduction

        self.ohem = ohem
        self.top_k_ratio = top_k_ratio
        print("#Train with BCEWithLogitsLoss\n")

    def forward(self, pred, target):
        # pred - (N,1,H,W); target - (N,H,W)
        N = target.shape[0]
        # (N,1,H,W)->(N,H*W); (N,H,W)->(N,H*W)
        # BCE要求预测和标签是float类型
        loss = super(BCE, self).forward(pred.squeeze(1).view(N, -1), target.view(N, -1).float())

        if self.ohem:
            assert loss.shape[0] == N and loss.ndim == 2
            # (N,H*W)->(N,)　每个样本的平均loss，这里BCE是对每个像素点进行计算的
            loss = loss.mean(dim=1)
            # OHEM模式下，只取loss最大的topk样本进行学习
            keep_num = max(1, int(self.top_k_ratio * N))
            top_k_loss, indices = loss.topk(keep_num)

            if self._reduction == 'mean':
                return top_k_loss.mean(), indices
            elif self._reduction == 'sum':
                return top_k_loss.sum(), indices
            elif self._reduction == 'none':
                return top_k_loss, indices
            else:
                raise NotImplementedError("Not implemented with reduction '{}'".format(self.reduction))
        else:
            indices = torch.arange(N)
            return loss, indices


class BCEDice:
    """Combine BCEWithLogitsLoss with DiceLoss, plus OHEM.
       pos_weight是正样本的bce loss权重，通常设置为正负样本比的倒数。"""

    def __init__(self, pos_weight=None, p=2, smooth=1e-8, reduction='mean', ohem=False, top_k_ratio=1.):
        self.bce = BCE(reduction=reduction, pos_weight=pos_weight, ohem=ohem, top_k_ratio=top_k_ratio)
        self.dice = Dice(p, smooth, reduction)
        # self.dice = Dice(p, smooth, reduction, ohem, top_k_ratio)
        print("#Train with BCEWithLogitsLoss & DiceLoss\n")

    def __call__(self, pred, target):
        # pred - (N,1,H,W); target - (N,H,W)
        bce_loss, bce_indices = self.bce(pred, target)
        dice_loss, dice_indices = self.dice(pred, target)
        loss = bce_loss + dice_loss
        # print("bce indices:", bce_indices)
        # print("dice indices:", dice_indices)

        return loss, bce_loss.detach(), dice_loss.detach(), bce_indices, dice_indices
