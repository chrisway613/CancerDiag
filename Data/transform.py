__all__ = ('PadResize', 'ToNormTensor', 'Scale', 'PILResize', 'ConvertToTensor', 'Norm',)

import torch
import torch.nn.functional as F

import numpy as np

from PIL import Image
from collections import Sequence

from torchvision.transforms import Compose, ToTensor, Normalize


class PadResize:
    """Resize an PIL Image to the target size, with unchanged aspect ratio using padding.
       note that this way will lose the edge info for segmentation"""

    def __init__(self, target_size: (int, tuple), interpolation=Image.BICUBIC):
        assert isinstance(target_size, (int, tuple))
        if isinstance(target_size, int):
            target_size = (target_size, target_size)

        self.size = target_size
        self.inter = interpolation

    def __call__(self, item: dict):
        img = item.get('image')
        resized_img = self._resize(img)

        if item.get('label'):
            mask = item.get('label')
            # mask使用最近邻插值？
            resized_mask = self._resize(mask, inter=Image.NEAREST)
            item.update(image=resized_img, label=resized_mask)
        else:
            item.update(image=resized_img)

        return item

    def _resize(self, img, inter=None):
        # 原始尺寸
        ori_w, ori_h = img.size
        # 目标尺寸
        tar_w, tar_h = self.size
        # 宽、高最小的缩放比例
        scale = min(tar_w / ori_w, tar_h / ori_h)
        # 被“有效”缩放的宽、高，保持原始的宽、高比
        val_w, val_h = int(scale * ori_w), int(scale * ori_h)

        if inter is None:
            inter = self.inter
        # 保持原图宽、高比进行缩放
        # 但注意这个尺寸不是目标尺寸
        resized_img = img.resize((val_w, val_h), inter)
        # float32类型
        valid = np.asarray(resized_img, dtype='float')

        img_arr = np.asarray(img, dtype='float')
        # 图像是3维矩阵
        if img_arr.ndim == 3:
            # 各通道像素值均值
            pad = img_arr.mean(axis=(0, 1))
            target_arr = np.zeros((tar_h, tar_w, 3))
        # mask是二维矩阵
        # 这种方法对mask不可行，会丢失边缘信息，且会使得mask非二值
        else:
            pad = img_arr.mean()
            # print("pad value:{}".format(pad))
            target_arr = np.zeros((tar_h, tar_w))
        target_arr[:, :] = pad

        # 中心区域维持原图宽、高比
        start_y = (tar_h - val_h) // 2
        end_y = start_y + val_h
        start_x = (tar_w - val_w) // 2
        end_x = start_x + val_w
        target_arr[start_y:end_y, start_x:end_x] = valid
        # print("unique target_arr:{}".format(np.unique(target_arr)))
        # 还原成图像时注意转换回uint8类型
        target_img = Image.fromarray(target_arr.astype('uint8'))
        # print("unique target uint8:{}".format(np.unique(target_img)))

        return target_img


class ToNormTensor:
    """Convert PIL Image to normalized tensor."""

    def __init__(self, mean: (Sequence, int, float) = None, std: (Sequence, int, float) = None):
        if mean is not None and std is not None:
            if not isinstance(mean, Sequence):
                mean = [mean] * 3
            if not isinstance(std, Sequence):
                std = [std] * 3

            for m in mean:
                assert 0. <= m <= 255.
                if m > 1:
                    m /= 255.
            for s in std:
                assert 0. <= s <= 255.
                if s > 1:
                    s /= 255.

        self.mean = mean
        self.std = std

    def __call__(self, item: dict):
        if self.mean is not None and self.std is not None:
            # Normalize()的操作对象是（C,H,W）的tensor，因此先使用ToTensor()
            # 但注意ToTensor()将对象归一化到0-1之间，因此这里mean和std都需要在0-1之间
            _transform = Compose([
                ToTensor(),
                Normalize(self.mean, self.std)
            ])
        else:
            _transform = ToTensor()

        img = item.get('image')
        img_tensor = _transform(img)
        item.update(image=img_tensor)

        if item.get('label') is not None:
            mask = item.get('label')
            # uint8->int64
            # 使用.copy()，否则torch会出现warning
            mask_tensor = torch.from_numpy(np.asarray(mask, dtype='long').copy())
            # # (H, W) -> (1, H, W)
            # # 为mask增加1个对应通道的维度
            # mask_tensor = mask_tensor.unsqueeze(0)
            # assert mask_tensor.dim() == 3 and mask_tensor.shape[0] == 1
            item.update(label=mask_tensor)

        return item


class ConvertToTensor:
    """Convert PIL Image to Pytorch tensor."""

    def __init__(self):
        self._transform = ToTensor()

    def __call__(self, item: dict):
        img = item.get('image')
        # uint8->float32
        img_tensor = self._transform(img)
        item.update(image=img_tensor)

        mask = item.get('label')
        if mask:
            # uint8->int64
            mask_tensor = torch.from_numpy(np.asarray(mask, dtype='long').copy())
            item.update(label=mask_tensor)

        return item


class Norm:
    """Normalize a tensor image with mean and standard deviation."""

    def __init__(self, mean=None, std=None):
        if mean is None or std is None:
            mean = [0.] * 3
            std = [1.] * 3

        for i, m in enumerate(mean):
            assert 0 <= m <= 255
            if m > 1:
                mean[i] = m / 255
        for j, s in enumerate(std):
            assert 0 <= s <= 255
            if s > 1:
                std[j] = s / 255

        # print("mean=", mean)
        # print("std=", std)

        self._transform = Normalize(mean, std)

    def __call__(self, item: dict):
        img = item.get('image')
        normalized_img = self._transform(img)
        item.update(image=normalized_img)

        return item


class Scale:
    """Scale the image tensor with Pytorch implementation.
       ps: recommend to set align_corners=True if scale the image, default is False"""

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        if mode in ('nearest', 'area') and align_corners:
            raise ValueError("align_corners option can only be set with the "
                             "interpolating modes: linear | bilinear | bicubic | trilinear")

        assert size is not None or scale_factor is not None, "'size' and 'scale factor' cannot be both None!"

        # 目标尺寸（高，宽）
        if size is not None:
            self.size = size
        # 缩放系数
        if scale_factor is not None:
            self.factor = scale_factor

        # 插值方式
        self.mode = mode
        # 像素是看作角点还是方形的中心
        self.align_corners = align_corners

    def __call__(self, item: dict):
        # Pytorch的插值方法对于图像需要input是4D张量
        # (H,W,C) -> (C,H,W)，注意转换为float，插值需要
        # img_arr = np.asarray(item.get('image'), dtype='float').transpose(2, 0, 1)
        # 3D->4D tensor: (C,H,W)->(1,C,H,W)
        # img_ts = torch.from_numpy(img_arr).unsqueeze(0)
        img_ts = item.get('image').unsqueeze(0)
        if self.size is not None:
            resized_img_ts = F.interpolate(img_ts, self.size, mode=self.mode, align_corners=self.align_corners)
        else:
            resized_img_ts = F.interpolate(img_ts, scale_factor=self.factor, mode=self.mode, align_corners=self.align_corners)

        # 4D->3D tensor: (1,C,H,W)->(C,H,W)
        img = resized_img_ts.squeeze(0)
        _, h, w = img.shape
        assert h == self.size[0] and w == self.size[1]
        # (C,H,W)->(H,W,C)
        # img_arr = img_ts.squeeze(0).numpy().transpose(1, 2, 0)
        # # 注意恢复为uint8
        # img = Image.fromarray(img_arr.astype('uint8'))
        item.update(image=img)

        if item.get('label') is not None:
            # mask使用最近邻插值
            # 插值时要求输入是浮点类型
            # (H,W)
            # mask_arr = np.asarray(item.get('label'), dtype='float')
            # (H,W)->(1,H,W)
            # mask_arr = mask_arr[np.newaxis].transpose(2, 0, 1)
            # 3D->4D tensor: (1,H,W)->(1,1,H,W)
            # mask_ts = torch.from_numpy(mask_arr).unsqueeze(0)
            mask_ts = item.get('label').float()
            mask_ts = mask_ts.unsqueeze(0).unsqueeze(0)
            # mask的使用默认的最近邻方式进行插值
            if self.size is not None:
                resized_mask_ts = F.interpolate(mask_ts, self.size)
            else:
                resized_mask_ts = F.interpolate(mask_ts, scale_factor=self.factor)

            # (1,1,H,W)->(H,W)
            # mask_arr = mask_ts.squeeze().numpy()
            # 恢复为uint8类型
            # mask = Image.fromarray(mask_arr.astype('uint8'))
            # float->long
            mask = resized_mask_ts.squeeze().long()
            h, w = mask.shape
            assert h == self.size[0] and w == self.size[1]
            item.update(label=mask)

        return item


class PILResize:
    """Resize the PIL image."""

    def __init__(self, size, mode=Image.BILINEAR):
        # (W,H)
        self.size = size
        self.mode = mode

    def __call__(self, item):
        image = item.get('image')
        resized_image = image.resize(self.size, self.mode)
        assert resized_image.size == self.size
        item.update(image=resized_image)

        mask = item.get('label')
        if mask:
            # mask使用最近邻插值
            resized_mask = mask.resize(self.size, Image.NEAREST)
            assert resized_mask.size == self.size
            item.update(label=resized_mask)

        return item
