import os
import numpy as np

from PIL import Image

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler
from torchvision.transforms import Compose

from Data.transform import Resize, ToNormTensor

# 训练集图像宽、高均值
TRAIN_SIZE = (839, 573)
# 测试集图像宽、高均值
TEST_SIZE = (512, 512)


class CancerDataset(Dataset):
    @classmethod
    def _encode_label(cls, raw_label):
        """对标签mask进行编码

            Args:
                raw_label (PIL Image): 将被编码的mask图像

            Returns:
                PIL Image: Encoded mask.
        """
        # uint8类型，保持与原图mask一致
        label_arr = np.asarray(raw_label, dtype='uint8').copy()
        label_arr[label_arr < 10] = 0
        label_arr[label_arr > 245] = 255
        # 阴性样本的mask全0，阳性样本为0或255
        assert len(np.unique(label_arr)) < 3, "len(np.unique(label_arr)) should below 2, got {}, details:{}".format(
            len(np.unique(label_arr)), np.unique(label_arr))

        return Image.fromarray(label_arr)

    def __init__(self, root, train=True, transform=None):
        super(CancerDataset, self).__init__()
        assert os.path.isdir(root), "'root' should be a parent directory which contains images and labels."

        # 图片文件所在目录
        self.image_dir = os.path.join(root, 'Images')
        assert os.path.exists(self.image_dir)
        # 图片文件路径
        self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]

        self.train = train
        if train:
            # 标签文件所在目录
            self.label_dir = os.path.join(root, 'Labels')
            assert os.path.exists(self.label_dir)
            # 标签文件路径
            self.label_paths = [os.path.join(self.label_dir, file) for file in os.listdir(self.label_dir)]

        # 数据格式转换、增强
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image_name = os.path.basename(image_path)
        image = Image.open(image_path)
        # (宽，高)
        size = image.size

        if self.train:
            label_path = self.label_paths[index]
            label_name = os.path.basename(label_path)
            label = Image.open(label_path)
            # 将标签编码为两类，像素值仅有0和255，非黑即白
            label = self._encode_label(label)
            item = dict(image=image, label=label, image_name=image_name, image_size=size, label_name=label_name)
        else:
            item = dict(image=image, image_name=image_name, image_size=size)

        if self.transform:
            item = self.transform(item)

        return item


if __name__ == '__main__':
    '-----------------------------------Data Loading Pipeline----------------------------------------'
    # ds = CancerDataset('Train')
    # print("Total {} images".format(len(ds)))
    #
    # for data in ds:
    #     image, label, image_name, image_size, label_name = data.values()
    #     print(image.size)
    #     print(image_size)
    #     print(image_name, label_name)
    #
    #     image.show()
    #     label.show()
    #
    #     image_arr = np.asarray(image, dtype='float')
    #     label_arr = np.asarray(label, dtype='uint8')
    #     # (H, W, C)
    #     print(image_arr.shape)
    #     # 阴性样本为0，阳性样本为0和255
    #     print(np.unique(label_arr))
    #
    #     rs = Resize(TRAIN_SIZE)
    #     rs_data = rs(data)
    #     resized_image = rs_data.get('image')
    #     resized_label = rs_data.get('label')
    #     print(resized_image.size, resized_label.size)
    #     resized_image.show()
    #     resized_label.show()
    #
    #     break
    #
    # # sample indices
    # rand_sampler = RandomSampler(ds)
    # bt_sampler = BatchSampler(rand_sampler, 4, True)
    # print("Total {} batches, batch size is {}".format(len(bt_sampler), bt_sampler.batch_size))
    #
    # for sample in bt_sampler:
    #     print(sample)
    #     break

    '----------------------------------------W Data Transformation-------------------------------------------'

    ds_transform = CancerDataset(
        'Train',
        transform=Compose([
            Resize(TRAIN_SIZE),
            ToNormTensor(mean=180.881, std=63.776)
        ]))
    dl_transform = DataLoader(ds_transform, batch_size=4, shuffle=True, drop_last=True)
    for batch_data in dl_transform:
        batch_images, batch_labels, \
            batch_image_names, batch_image_sizes, batch_label_names = batch_data.values()

        # (batch, channel, H, W)
        print(batch_images.shape, batch_labels.shape)
        # float32, uint8
        print(batch_images.dtype, batch_labels.dtype)

        print(batch_image_names)
        # 返回类似：[tensor([867, 954, 939, 942]), tensor([580, 633, 621, 625])]
        # 前一组tensor是原图的宽、后一组是原图的高
        print(batch_image_sizes)
        print(batch_label_names)

        break
