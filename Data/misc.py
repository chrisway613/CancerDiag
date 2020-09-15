import os

from sklearn.model_selection import train_test_split

from torchvision.transforms import Compose
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler

from conf import *
from .load_data import CancerDataset
from .transform import ConvertToTensor, Scale, Norm


def train_eval_split():
    """从训练集中按一定比例划分出验证集，且要求划分后训练集中阳性样本数量占阴性样本数量的７成以上"""

    # 训练集目录
    train_dir = os.path.join(DATA_DIR, 'Train')
    # 训练集图像目录
    image_dir = os.path.join(train_dir, 'Images')
    # 训练集标签目录
    label_dir = os.path.join(train_dir, 'Labels')
    # 训练图像路径
    image_paths = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]
    # 训练标签图像路径
    label_paths = [os.path.join(label_dir, file) for file in os.listdir(label_dir)]

    train_img_paths = train_label_paths = None
    eval_img_paths = eval_label_paths = None

    train_rate = 0.
    # 当阳性样本数量不足阴性样本的七成时重新划分
    # 因为训练集中阴性样本数量:阳性样本数量就是约为10:7
    while train_rate < .7:
        # 从训练集中按比例划分验证集
        train_img_paths, eval_img_paths, \
            train_label_paths, eval_label_paths = train_test_split(image_paths, label_paths,
                                                                   test_size=SPLIT_RATIO, shuffle=True)
        print("num of images in training set:{}, num of images in validation set:{}".format(
            len(train_img_paths), len(eval_img_paths)))

        # 划分后训练集中阳性样本数量
        train_pos_num = len([path for path in train_img_paths if int(os.path.basename(path).split('.')[0]) > 1000])
        # 划分后训练集中阴性样本数量
        train_neg_num = len(train_img_paths) - train_pos_num
        # 划分后训练集中阳性样本/阴性样本
        train_rate = train_pos_num / train_neg_num if train_neg_num else 0.
        print("[Train] positive samples num:{}, negative samples num:{}, rate(pos/neg)={:.3f}".format(
            train_pos_num, train_neg_num, train_rate))

        # 划分后验证集中阳性样本数量
        eval_pos_num = len([path for path in eval_img_paths if int(os.path.basename(path).split('.')[0]) > 1000])
        # 划分后验证集中阴性样本数量
        eval_neg_num = len(eval_img_paths) - eval_pos_num
        # 划分后验证集中阳性样本/阴性样本
        eval_rate = eval_pos_num / eval_neg_num if eval_neg_num else 0.
        print("[Eval] positive samples num:{}, negative samples num:{}, rate(pos/neg)={:.3f}".format(
            eval_pos_num, eval_neg_num, eval_rate))

    return train_img_paths, eval_img_paths, train_label_paths, eval_label_paths


def draw_tiny_samples(weights, num, bs):
    """从数据集中按照一定比重采样少量样本"""

    ds = CancerDataset(root=os.path.join(DATA_DIR, 'Train'), transform=Compose([
        ConvertToTensor(),
        Scale(size=(INPUT_SIZE[1], INPUT_SIZE[0]), mode='bilinear', align_corners=True),
        Norm(mean=TRAIN_SCALE_MEAN, std=TRAIN_SCALE_STD)
    ]))

    # 带权重的随机采样，replacement=False代表不会重复采同一个样本
    # 注意sampler中代表的是Dataset的索引
    # 前1000个是阴性样本，后732个是阳性样本，采样比重10:7
    wt_rand_sampler = WeightedRandomSampler(weights, len(ds), replacement=False)
    # 采一定数量的样本
    indices = []
    for sample in wt_rand_sampler:
        # print(sample)
        indices.append(sample)
        if len(indices) == num:
            break
    assert len(indices) == num

    # 将以上样本索引包装为sampler
    sub_rand_sampler = SubsetRandomSampler(indices)
    assert len(sub_rand_sampler) == len(indices)

    # bt_sampler = BatchSampler(sub_rand_sampler, 4, True)
    # print("Total {} batches".format(len(bt_sampler)))
    # dl = DataLoader(ds, batch_sampler=bt_sampler)
    # assert len(bt_sampler) == len(dl)

    dl = DataLoader(ds, batch_size=bs, sampler=sub_rand_sampler, drop_last=True)
    # for batch_data in dl:
    #     batch_images, batch_labels, batch_image_names, batch_image_sizes, batch_label_names = batch_data.values()
    #     print(batch_images.shape, batch_labels.shape)
    #     print(batch_image_names, batch_label_names)
    #     print(batch_image_sizes)
    print("Draw {} batches of {} samples".format(len(dl), num))

    return dl
