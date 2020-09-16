__all__ = ('UNet',)

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnetConvBlock(nn.Module):
    """Base Convolution Block of U-Net."""

    def __init__(self, in_planes, planes, padding: bool = False):
        super(UnetConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, planes, 3, padding=int(padding), bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Conv2d(planes, planes, 3, padding=int(padding), bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_planes, planes, mode='interpolation', padding: bool = False):
        assert mode in ('interpolation', 'convolution')
        super(UpBlock, self).__init__()

        self.mode = mode
        if mode == 'convolution':
            self.up_conv = nn.ConvTranspose2d(in_planes, planes, 2, stride=2)
        else:
            self.pw_conv = nn.Conv2d(in_planes, planes, 1)

        # 使用1x1conv则不改变尺寸
        # self.conv = nn.Conv2d(in_planes, planes, 1)
        # 这里边长会减少4
        self.conv = UnetConvBlock(in_planes, planes, padding)

    def _crop(self, f, size):
        ori_h, ori_w = f.shape[-2:]
        h, w = size

        start_y = (ori_h - h) // 2
        start_x = (ori_w - w) // 2
        end_y = start_y + h
        end_x = start_x + w

        # (h,w)
        return f[:, :, start_y:end_y, start_x:end_x]

    def forward(self, x, mid):
        if self.mode == 'interpolation':
            size = mid.shape[-2:]
            # # 插值到低层特征图的尺寸
            # x = F.interpolate(x, size, mode='bilinear', align_corners=True)
            # 上采样2倍
            x = F.interpolate(x, scale_factor=2., mode='bilinear', align_corners=True)
            # 再将通道数映射到与低层特征图一致
            x = self.pw_conv(x)
            # 将低层特征图裁剪到和高层上采样后的尺寸一致
            mid = self._crop(mid, x.shape[-2:])
        else:
            # 上采样2倍
            x = self.up_conv(x)
            # 将低层特征图裁剪到和高层上采样后的尺寸一致
            mid = self._crop(mid, x.shape[-2:])

        # 在通道维度上拼接
        fusion = torch.cat([mid, x], dim=1)
        # 拼接后通道数翻倍，因此最后再经过卷积块压缩
        out = self.conv(fusion)

        return out


class UNet(nn.Module):
    def __init__(self, in_planes, planes, mode='interpolation', padding: bool = False, depth=5):
        assert mode in ('interpolation', 'convolution')
        super(UNet, self).__init__()

        # 下采样模块
        self.down_path = nn.ModuleList()
        in_channels = in_planes
        for i in range(depth):
            # 64 -- 1024
            out_channels = 2 ** (6 + i)
            self.down_path.append(UnetConvBlock(in_channels, out_channels, padding))
            in_channels = out_channels

        # 上采样模块
        self.up_path = nn.ModuleList()
        for j in reversed(range(depth - 1)):
            # 512 -- 64
            out_channels = 2 ** (6 + j)
            self.up_path.append(UpBlock(in_channels, out_channels, mode, padding))
            in_channels = out_channels

        # 1x1卷积映射通道
        self.proj = nn.Conv2d(in_channels, planes, 1)

    def forward(self, x):
        # 先记录下输入图像尺寸（H,W）
        size = x.shape[-2:]
        # print("Input size:{}".format(x.shape))

        # 下采样过程中间层输出
        mid_layers = []
        for i, down_block in enumerate(self.down_path):
            x = down_block(x)
            # print("output size of down block{}: {}".format(i, x.shape))
            if i != len(self.down_path) - 1:
                mid_layers.append(x)
                x = F.max_pool2d(x, 2, stride=2)
                # print("output size by pooling{}: {}".format(i, x.shape))

        for j, up_block in enumerate(self.up_path):
            mid = mid_layers[-1 - j]
            x = up_block(x, mid)
            # print("output size by up block{}: {}".format(j, x.shape))

        # 将通道映射到类别数量
        x = self.proj(x)
        # 插值回输入图像尺寸大小
        out = F.interpolate(x, size, mode='bilinear', align_corners=True)
        # print("final output size:{}".format(out.shape))

        return out


if __name__ == '__main__':
    net = UNet(3, 2)
    img = torch.randn(1, 3, 744, 512)
    out = net(img)
