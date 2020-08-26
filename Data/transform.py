__all__ = ('Resize', 'ToNormTensor',)

import torch
import numpy as np

from PIL import Image
from collections import Sequence

from torchvision.transforms import Compose, ToTensor, Normalize


class Resize:
    """Resize an PIL Image to the target size, with unchanged aspect ratio using padding."""

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
        # 数据集中有部分mask不是3通道，需要进行转换
        if img.mode != 'RGB':
            img = img.convert('RGB')

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
        resized_img = img.resize((val_w, val_h), inter)
        # float32类型
        valid = np.asarray(resized_img, dtype='float32')

        img_arr = np.asarray(img, dtype='float32')
        # 周边用原图各通道像素均值填充
        pad = img_arr.mean(axis=(0, 1))
        target_arr = np.zeros((tar_h, tar_w, 3))
        target_arr[:, :] = pad

        # 中心区域维持原图宽、高比
        start_y = (tar_h - val_h) // 2
        end_y = start_y + val_h
        start_x = (tar_w - val_w) // 2
        end_x = start_x + val_w
        target_arr[start_y:end_y, start_x:end_x] = valid
        target_img = Image.fromarray(target_arr.astype('uint8'))

        return target_img


class ToNormTensor:
    """Convert PIL Image to normalized tensor."""

    def __init__(self, mean: (Sequence, int, float) = None, std: (Sequence, int, float) = None):
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
            _transform = Compose([
                ToTensor(),
                Normalize(self.mean, self.std)
            ])
        else:
            _transform = ToTensor()

        img = item.get('image')
        img_tensor = _transform(img)
        item.update(image=img_tensor)

        if item.get('label'):
            mask = item.get('label')
            # uint8->int32
            # 使用.copy()，否则torch会出现warning
            mask_tensor = torch.from_numpy(np.asarray(mask, dtype='int').copy())
            # (H, W, C) -> (C, H , W)
            mask_tensor = mask_tensor.permute(2, 0, 1)
            item.update(label=mask_tensor)

        return item
