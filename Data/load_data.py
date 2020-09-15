__all__ = ('CancerDataset',)

import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler, \
    SubsetRandomSampler, WeightedRandomSampler
from torchvision.transforms import Compose, ToPILImage

from Data.transform import PILResize, Scale, ConvertToTensor, Norm

# 图像缩放尺寸--宽、高
from conf import INPUT_SIZE

from PIL import Image
# PIL默认会检查解压炸弹DOS攻击，由于某些阳性样本尺寸太大，因此这里忽略检查
Image.MAX_IMAGE_PIXELS = None


class CancerDataset(Dataset):
    @classmethod
    def _encode_label(cls, raw_label):
        """对标签mask进行编码

            Args:
                raw_label (PIL Image): 将被编码的mask图像，共有0-9、246-255 20种像素值

            Returns:
                PIL Image: Encoded mask.
        """

        label_arr = np.asarray(raw_label).copy()
        assert label_arr.dtype == np.uint8

        # 编码为二值图
        label_arr[label_arr < 10] = 0
        label_arr[label_arr > 245] = 1
        # 阴性样本的mask全0，阳性样本为0或1
        assert len(np.unique(label_arr)) < 3, "len(np.unique(label_arr)) should below 2, got {}, details:{}".format(
            len(np.unique(label_arr)), np.unique(label_arr))

        return Image.fromarray(label_arr)

    def __init__(self, root=None, img_paths=None, label_paths=None, train=True, transform=None):
        """
            Args:
                root: parent directory which contains images & labels;
                img_paths: a sequence which contains all the path of image files;
                label_paths: a sequence which contains all the path of label files;
                train: indicate training set or not;
                transform: data augmentation

            Note that we use 'img_paths' & 'label_paths' only if 'root' is None.
        """

        super(CancerDataset, self).__init__()
        assert root is not None or (img_paths is not None and label_paths is not None)

        if root:
            assert os.path.isdir(root), "'root' should be a parent directory which contains images and labels."

            self.image_dir = os.path.join(root, 'Images')
            assert os.path.exists(self.image_dir)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]

            if train:
                self.label_dir = os.path.join(root, 'Labels')
                assert os.path.exists(self.label_dir)
                self.label_paths = [os.path.join(self.label_dir, file) for file in os.listdir(self.label_dir)]
        else:
            self.image_dir = os.path.dirname(img_paths[0])
            assert os.path.exists(self.image_dir)
            self.image_paths = img_paths

            if train:
                self.label_dir = os.path.dirname(label_paths[0])
                assert os.path.exists(self.label_dir)
                self.label_paths = label_paths

        assert os.path.dirname(self.label_dir) == os.path.dirname(self.image_dir)

        self.image_paths = sorted(self.image_paths, key=lambda p: int(os.path.basename(p).split('.')[0]))
        self.label_paths = sorted(self.label_paths, key=lambda p: int(os.path.basename(p).split('_mask.')[0]))
        assert len(self.image_paths) == len(self.label_paths)

        self.train = train
        # 数据格式转换、增强
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # 支持切片
        if isinstance(index, slice):
            self.image_paths = self.image_paths[index]
            self.label_paths = self.label_paths[index]

            return self

        image_path = self.image_paths[index]
        image_name = os.path.basename(image_path)
        image = Image.open(image_path)
        # (宽，高)
        size = image.size

        if self.train:
            label_path = self.label_paths[index]
            label_name = os.path.basename(label_path)
            label = Image.open(label_path)
            assert label.mode == 'L'
            # 将标签编码为两类，0和1，对应解码为0和255，二值图，非黑即白
            label = self._encode_label(label)
            item = dict(image=image, label=label, image_name=image_name, image_size=size, label_name=label_name)
        else:
            item = dict(image=image, image_name=image_name, image_size=size)

        if self.transform:
            item = self.transform(item)

        return item


if __name__ == '__main__':
    '-----------------------------------Data Loading Pipeline----------------------------------------'
    ds = CancerDataset(root='Train')
    print("Total {} images".format(len(ds)))

    for data in reversed(ds[-3:-10:-1]):
        image, label, image_name, image_size, label_name = data.values()
        # （宽，高）
        print(image.size)
        print(image_size)
        print(image_name, label_name)
        # RGB、L
        print(image.mode, label.mode)

        image.show()
        label_arr = np.asarray(label).copy()
        # 解码
        label_arr *= 255
        label = Image.fromarray(label_arr)
        label.show()

        # 图像使用float32类型
        image_arr = np.asarray(image, dtype='float')
        # mask使用int64类型
        label_arr = np.asarray(label, dtype='long')
        # (H, W, 3)
        print(image_arr.shape)
        # (H, W)
        print(label_arr.shape)
        # 阴性样本为0，阳性样本为0和255
        print(np.unique(label_arr))

        pil_resize = PILResize(INPUT_SIZE)
        resized_data = pil_resize(data)

        resized_image = resized_data.get('image')
        # RGB
        print(resized_image.mode)
        print(resized_image.size)
        resized_image.show()

        resized_mask = resized_data.get('label')
        # L
        print(resized_mask.mode)
        resized_mask_arr = np.asarray(resized_mask).copy()
        # uint8
        print(resized_mask_arr.dtype)
        # 0,1
        print(np.unique(resized_mask_arr))
        # 解码
        resized_mask_arr *= 255
        resized_mask = Image.fromarray(resized_mask_arr)
        resized_mask.show()

        break

    # sample indices
    rand_sampler = RandomSampler(ds)
    bt_sampler = BatchSampler(rand_sampler, 4, True)
    print("Total {} batches, batch size is {}".format(len(bt_sampler), bt_sampler.batch_size))

    for sample in bt_sampler:
        print(sample)
        break

    ds = CancerDataset(root='Train', transform=Compose([
        # 转换成张量
        ConvertToTensor(),
        # 缩放到统一尺寸
        Scale(INPUT_SIZE, mode='bilinear', align_corners=True)
    ]))

    # 带权重的随机采样，replacement=False代表不会重复采同一个样本
    # 注意sampler中代表的是Dataset的索引
    # 前1000个是阴性样本，后732个是阳性样本，采样比重10:7
    wt_rand_sampler = WeightedRandomSampler([1.] * 1000 + [.7] * 732, len(ds), replacement=False)
    # 采10个样本
    num = 10
    indices = []
    for sample in wt_rand_sampler:
        print(sample)
        indices.append(sample)

        if len(indices) == num:
            break
    assert len(indices) == 10

    # 将以上10个样本索引包装为sampler
    sub_rand_sampler = SubsetRandomSampler(indices)
    assert len(sub_rand_sampler) == len(indices)

    # bt_sampler = BatchSampler(sub_rand_sampler, 4, True)
    # print("Total {} batches".format(len(bt_sampler)))
    # dl = DataLoader(ds, batch_sampler=bt_sampler)
    # assert len(bt_sampler) == len(dl)

    dl = DataLoader(ds, batch_size=4, sampler=sub_rand_sampler, drop_last=True)
    for batch_data in dl:
        batch_images, batch_labels, batch_image_names, batch_image_sizes, batch_label_names = batch_data.values()
        print(batch_images.shape, batch_labels.shape)
        print(batch_image_names, batch_label_names)
        print(batch_image_sizes)

    # '----------------------------------------W Data Transformation-------------------------------------------'

    W, H = INPUT_SIZE
    MEAN = [208.644, 184.249, 206.240]
    STD = [54.267, 77.503, 51.150]

    ds_transform = CancerDataset(
        root='Train',
        transform=Compose([
            # PILResize(SIZE),
            # ToNormTensor(mean=MEAN, std=STD),
            ConvertToTensor(),
            Scale(size=(H, W), mode='bilinear', align_corners=True),
            Norm(mean=[208.644, 184.249, 206.240], std=[54.267, 77.503, 51.150])
        ]))
    dl_transform = DataLoader(ds_transform, batch_size=4, shuffle=True, drop_last=True)

    for batch_data in dl_transform:
        batch_images, batch_labels, \
            batch_image_names, batch_image_sizes, batch_label_names = batch_data.values()

        # (batch, 3, H, W), (batch, H, W)
        print(batch_images.shape, batch_labels.shape)
        # float32, int64
        print(batch_images.dtype, batch_labels.dtype)
        print(batch_image_names)
        # 返回类似：[tensor([867, 954, 939, 942]), tensor([580, 633, 621, 625])]
        # 前一组tensor是原图的宽、后一组是原图的高
        print(batch_image_sizes)
        print(batch_label_names)

        # 显示缩放后的mask
        for label_ts in batch_labels:
            label_arr = label_ts.numpy().astype('uint8')
            label_arr *= 255
            mask = Image.fromarray(label_arr)
            mask.show()

        # 显示缩放后的图像
        for image_ts in batch_images:
            # (C,H,W)->(H,W,C)
            image_ts = image_ts.permute(1, 2, 0)
            # print(image_ts.shape)
            # 反归一化
            image_ts.mul_(torch.as_tensor(STD, dtype=image_ts.dtype)).add_(torch.as_tensor(MEAN, dtype=image_ts.dtype))
            image_arr = image_ts.numpy().astype('uint8')
            image = Image.fromarray(image_arr)
            print(image.size)
            image.show()

        # 还原尺寸后显示
        for image_ts, label_ts, image_w, image_h in zip(batch_images, batch_labels, batch_image_sizes[0], batch_image_sizes[1]):
            # (C,H,W)->(H,W,C)
            image_ts = image_ts.permute(1, 2, 0)
            # 反归一化
            # image_ts.mul_(torch.as_tensor(STD, dtype=image_ts.dtype)).add_(torch.as_tensor(MEAN, dtype=image_ts.dtype))
            image_ts.mul_(
                torch.as_tensor(STD, dtype=image_ts.dtype) / 255
            ).add_(
                torch.as_tensor(MEAN, dtype=image_ts.dtype) / 255
            )
            # (H,W,C)->(C,H,W)
            image_ts = image_ts.permute(2, 0, 1)
            # (H,W)
            # label_ts.mul_(255)

            scale = Scale(size=(image_h, image_w), mode='bilinear', align_corners=True)
            rec_data = scale(dict(image=image_ts, label=label_ts))
            # float, long
            rec_image_ts, rec_label_ts = rec_data.values()
            # mask解码：0、1->0、255
            rec_label_ts *= 255

            # (C,H,W)->(H,W,C)
            # rec_image_arr = rec_image_ts.numpy().astype('uint8').transpose(1, 2, 0)
            # rec_image = Image.fromarray(rec_image_arr)
            to_pil = ToPILImage()
            rec_image = to_pil(rec_image_ts)
            print(rec_image.size, rec_image.mode)

            rec_label_arr = rec_label_ts.numpy().astype('uint8')
            rec_label = Image.fromarray(rec_label_arr)
            assert rec_image.size == rec_label.size, print(rec_label.size)

            rec_image.show()
            rec_label.show()

        break
