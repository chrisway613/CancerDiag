import os
import math
import random

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
    # 排序！
    image_paths = sorted(image_paths, key=lambda p: int(os.path.basename(p).split('.')[0]))
    # 训练标签图像路径
    label_paths = [os.path.join(label_dir, file) for file in os.listdir(label_dir)]
    # 排序！
    label_paths = sorted(label_paths, key=lambda p: int(os.path.basename(p).split('_mask.')[0]))

    assert len(image_paths) == len(label_paths)

    # 随机数种子
    if os.path.exists(RANDOM_SEED):
        with open(RANDOM_SEED, 'r') as f:
            random_seed = int(f.read().strip())
            print("Using random seed:{}".format(random_seed))
    else:
        random_seed = -1
        print("Initialize random seed={}".format(random_seed))

    train_rate = eval_rate = 0.
    # 当划分后训练集阳性样本数量不足阴性样本的七成或者验证集中没有阳性样本时重新划分
    # 因为原训练集中阴性样本数量:阳性样本数量就是约为10:7
    while train_rate < .7 or eval_rate < .5:
        if not os.path.exists(RANDOM_SEED):
            random_seed += 1

        # 从训练集中按比例划分验证集
        train_img_paths, eval_img_paths, \
            train_label_paths, eval_label_paths = train_test_split(image_paths, label_paths,
                                                                   test_size=SPLIT_RATIO, shuffle=True,
                                                                   random_state=random_seed)
        # print("num of images in training set:{}, num of images in validation set:{}".format(
        #     len(train_img_paths), len(eval_img_paths)))

        # 划分后训练集中的阳性样本
        train_img_pos_paths = [path for path in train_img_paths
                               if int(os.path.basename(path).split('.')[0]) > 1000]
        # train_label_pos_paths = [path for path in train_label_paths
        #                          if int(os.path.basename(path).split('_mask.')[0]) > 1000]
        # 划分后训练集中阳性样本数量
        train_pos_num = len(train_img_pos_paths)
        # 划分后训练集中阴性样本数量
        train_neg_num = len(train_img_paths) - train_pos_num
        # 划分后训练集中阳性样本/阴性样本
        # 此处可控制训练集是否要包含阴性样本
        train_rate = train_pos_num / train_neg_num if train_neg_num else 0.  # math.inf
        # print("[Train] positive samples num:{}, negative samples num:{}, rate(pos/neg)={:.3f}".format(
        #     train_pos_num, train_neg_num, train_rate))

        # 划分后验证集中的阳性样本
        eval_img_pos_paths = [path for path in eval_img_paths
                              if int(os.path.basename(path).split('.')[0]) > 1000]
        eval_label_pos_paths = [path for path in eval_label_paths
                                if int(os.path.basename(path).split('_mask.')[0]) > 1000]
        assert len(eval_img_pos_paths) == len(eval_label_pos_paths)
        # 划分后验证集中阳性样本数量
        eval_pos_num = len(eval_img_pos_paths)
        # 划分后验证集中阴性样本数量
        eval_neg_num = len(eval_img_paths) - eval_pos_num
        # 划分后验证集中阳性样本/阴性样本
        eval_rate = eval_pos_num / eval_neg_num if eval_neg_num else math.inf
        # print("[Eval] positive samples num:{}, negative samples num:{}, rate(pos/neg)={:.3f}".format(
        #     eval_pos_num, eval_neg_num, eval_rate))

    # 将随机数种子写入文件，保证下次划分数据集得到相同的结果
    with open(RANDOM_SEED, 'w') as f:
        f.write(str(random_seed))
        print("overwritten random seed={}\n".format(random_seed))

    print("[Train] positive samples num:{}, negative samples num:{}, rate(pos/neg)={:.3f}".format(
            train_pos_num, train_neg_num, train_rate))
    print("[Eval] positive samples num:{}, negative samples num:{}, rate(pos/neg)={:.3f}".format(
        eval_pos_num, eval_neg_num, eval_rate))

    # 令验证集只包含阳性样本
    eval_img_paths = eval_img_pos_paths
    eval_label_paths = eval_label_pos_paths
    print("#[Eval] drop negative samples\n")
    # # 只训练阳性样本
    # train_img_paths = train_img_pos_paths
    # train_label_paths = train_label_pos_paths

    # 检查图像与标签是否一一对应
    for img_path, label_path in zip(train_img_paths, train_label_paths):
        assert os.path.basename(img_path).split('.')[0] == os.path.basename(label_path).split('_mask.')[0]
    for img_path, label_path in zip(eval_img_paths, eval_label_paths):
        assert os.path.basename(img_path).split('.')[0] == os.path.basename(label_path).split('_mask.')[0]

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
    for i, sample in enumerate(wt_rand_sampler):
        # print(sample)
        if not indices:
            indices.append(sample)
        # 轮流采样阴阳样本，保证类别比例均衡
        if indices[-1] < 1000 <= sample:
            indices.append(sample)
        elif indices[-1] >= 1000 > sample:
            indices.append(sample)
        else:
            pass

        # 数量足够则结束采样
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
    print("Draw {} batches of {} samples\n".format(len(dl), num))

    return dl


def balanced_sample(img_paths, label_paths):
    """将全部阳性样本的路径保留，然后和阴性样本轮流加入到list中，直至阳性样本全部加入（多余的阴性样本抛弃）"""
    assert len(img_paths) == len(label_paths)

    # 分离阴阳样本的路径
    img_pos_paths, img_neg_paths = [], []
    for img_path, label_path in zip(img_paths, label_paths):
        # 检查图像与标签是否对应
        assert os.path.basename(img_path).split('.')[0] == os.path.basename(label_path).split('_mask.')[0]

        if int(os.path.basename(img_path).split('.')[0]) > 1000:
            img_pos_paths.append(img_path)
        else:
            img_neg_paths.append(img_path)

    balanced_img_paths, balanced_label_paths = [], []
    min_len = len(img_pos_paths) \
        if len(img_pos_paths) < len(img_neg_paths) else len(img_neg_paths)
    while min_len:
        # 随机抽取一个不重复的阳性样本路径
        img_pos_path = random.choice(img_pos_paths)
        while img_pos_path in balanced_img_paths:
            img_pos_path = random.choice(img_pos_paths)
        balanced_img_paths.append(img_pos_path)
        if len(img_pos_paths) < len(img_neg_paths):
            min_len -= 1

        # 获取对应样本的标签路径
        label_dir = os.path.dirname(img_pos_path).replace('Images', 'Labels')
        label_name = str(os.path.basename(img_pos_path).split('.')[0]) + '_mask.jpg'
        label_pos_path = os.path.join(label_dir, label_name)
        assert label_pos_path in label_paths, "'{}' not in label paths!".format(label_pos_path)
        balanced_label_paths.append(label_pos_path)

        # 随机抽取一个不重复的阴性样本路径
        img_neg_path = random.choice(img_neg_paths)
        while img_neg_path in balanced_img_paths:
            img_neg_path = random.choice(img_neg_paths)
        balanced_img_paths.append(img_neg_path)
        if len(img_neg_paths) <= len(img_pos_paths):
            min_len -= 1

        # 获取对应样本的标签路径
        label_dir = os.path.dirname(img_neg_path).replace('Images', 'Labels')
        label_name = str(os.path.basename(img_neg_path).split('.')[0]) + '_mask.jpg'
        label_neg_path = os.path.join(label_dir, label_name)
        assert label_neg_path in label_paths, "'{}' not in label paths!".format(label_neg_path)
        balanced_label_paths.append(label_neg_path)

    # 若还有未采样的阳性样本，则全部加入
    for path in set(img_pos_paths).difference(set(balanced_img_paths)):
        balanced_img_paths.append(path)

        # 将对应样本的标签路径也加入
        label_dir = os.path.dirname(path).replace('Images', 'Labels')
        label_name = str(os.path.basename(path).split('.')[0]) + '_mask.jpg'
        label_path = os.path.join(label_dir, label_name)
        assert label_path in label_paths, "'{}' not in label paths!".format(label_path)
        balanced_label_paths.append(label_path)

    # 阳性样本至少占一半
    assert len(img_pos_paths) >= len(balanced_img_paths) // 2
    # 必须包含所有阳性样本
    assert set(img_pos_paths).issubset(set(balanced_img_paths))
    # 图像与标注数量要对应
    assert len(balanced_img_paths) == len(balanced_label_paths)

    return balanced_img_paths, balanced_label_paths


def load_balanced_data(train_img_paths, eval_img_paths, train_label_paths, eval_label_paths):
    # 将阴阳样本的路径依次排列到列表中，用于构建Dataset，以保证之后每个batch内类别均衡
    train_img_paths, train_label_paths = balanced_sample(train_img_paths, train_label_paths)
    # eval_img_paths, eval_label_paths = balanced_sample(eval_x_paths, eval_y_paths)

    # 按采样好的样本顺序构建Dataset
    train_ds = CancerDataset(
        img_paths=train_img_paths, label_paths=train_label_paths,
        transform=Compose([
            ConvertToTensor(),
            Scale(size=(INPUT_SIZE[1], INPUT_SIZE[0]), mode='bilinear', align_corners=True),
            # Scale(size=(TINY_SIZE[1], TINY_SIZE[0]), mode='bilinear', align_corners=True),
            Norm(mean=TRAIN_SCALE_MEAN, std=TRAIN_SCALE_STD)
        ]),
        # 不排序，因为要按照采样好的顺序，阴阳样本依次相邻
        sort=False
    )
    eval_ds = CancerDataset(
        img_paths=eval_img_paths, label_paths=eval_label_paths,
        transform=Compose([
            ConvertToTensor(),
            Scale(size=(INPUT_SIZE[1], INPUT_SIZE[0]), mode='bilinear', align_corners=True),
            # Scale(size=(TINY_SIZE[1], TINY_SIZE[0]), mode='bilinear', align_corners=True),
            Norm(mean=EVAL_SCALE_MEAN, std=EVAL_SCALE_STD)
        ]),
        # 不排序，按照采样好的顺序，阴阳样本依次相邻
        sort=False
    )

    assert train_ds[0].get('image').shape[1:] == (INPUT_SIZE[1], INPUT_SIZE[0])
    assert eval_ds[1].get('image').shape[1:] == (INPUT_SIZE[1], INPUT_SIZE[0])
    print("Train with image size(width, height): {}".format(INPUT_SIZE))
    # assert train_ds[0].get('image').shape[1:] == (TINY_SIZE[1], TINY_SIZE[0])
    # assert eval_ds[1].get('image').shape[1:] == (TINY_SIZE[1], TINY_SIZE[0])
    return train_ds, eval_ds
