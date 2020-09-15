import os
import sys
import time

import torch
import torch.nn as nn

from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from conf import *
from utils import *
from models import *
from Data.misc import train_eval_split, draw_tiny_samples
# from data_process import calc_train_eval_pixel, calc_scale_pixel

from Data.load_data import CancerDataset
from Data.transform import ConvertToTensor, Norm, Scale

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(device_id) for device_id in DEVICE_ID])


def load_data():
    """Build Dataloader of training & validation set."""

    # if GPU:
    #     dev = torch.device('cuda:{}'.format(DEVICE_ID[0]))
    # else:
    #     dev = torch.device('cpu')

    # 从总体训练集中划分一部分作为验证集
    train_img_paths, eval_img_paths, train_label_paths, eval_label_paths = train_eval_split()
    # calc_train_eval_pixel(train_img_paths, eval_img_paths)

    # 构造训练集
    train_ds = CancerDataset(
        img_paths=train_img_paths,
        label_paths=train_label_paths,
        transform=Compose([
            # PILResize(INPUT_SIZE, mode=Image.BILINEAR),
            # 转换为0-1张量
            ConvertToTensor(),
            # 尺寸缩放，短边（高）缩放到512，同时保持基于训练集图像统计的宽高比均值
            # Scale(size=(INPUT_SIZE[1], INPUT_SIZE[0]), mode='bicubic', align_corners=True),
            Scale(size=(INPUT_SIZE[1], INPUT_SIZE[0]), mode='bilinear', align_corners=True),
            # 归一化
            Norm(mean=TRAIN_SCALE_MEAN, std=TRAIN_SCALE_STD)
        ])
    )
    # # 统计缩放后训练集图像的均值和标准差
    # calc_scale_pixel(train_ds, INPUT_SIZE, dev)

    # 构造验证集
    eval_ds = CancerDataset(
        img_paths=eval_img_paths,
        label_paths=eval_label_paths,
        transform=Compose([
            # PILResize(INPUT_SIZE, mode=Image.BILINEAR),
            # 转换为0-1张量
            ConvertToTensor(),
            # 尺寸缩放，短边（高）缩放到512，同时保持基于训练集图像统计的宽高比均值
            # Scale(size=(INPUT_SIZE[1], INPUT_SIZE[0]), mode='bicubic', align_corners=True),
            Scale(size=(INPUT_SIZE[1], INPUT_SIZE[0]), mode='bilinear', align_corners=True),
            # 归一化
            Norm(mean=TRAIN_SCALE_MEAN, std=TRAIN_SCALE_STD)
        ])
    )
    # # 统计缩放后验证集图像的均值和标准差
    # calc_scale_pixel(eval_ds, INPUT_SIZE, dev)

    # ds = CancerDataset(os.path.join(DATA_DIR, 'Train'), transform=Compose([
    #     # 缩放到基于数据集统计的尺寸均值
    #     # PILResize(INPUT_SIZE),
    #     # 转换为0-1张量并归一化
    #     # 均值和标准差都是基于数据集统计的值
    #     # ToNormTensor(mean=0, std=1)
    #     ConvertToTensor(),
    #     # 尺寸缩放，短边（高）缩放到512，同时保持基于训练集图像统计的宽高比均值
    #     Scale(size=INPUT_SIZE, mode='bilinear', align_corners=True)
    # ]))
    # # 按比例划分训练集和验证集
    # val_len = int(len(ds) * SPLIT_RATIO)
    # train_len = len(ds) - val_len
    # train_ds, val_ds = random_split(ds, (train_len, val_len))
    # print("num of images in training set:{}, num of images in validation set:{}".format(
    #     len(train_ds), len(val_ds)))
    #

    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    eval_dl = DataLoader(eval_ds, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    print("batch size={}, {} batches in training set, {} batches in validation set\n".format(
        BATCH_SIZE, len(train_dl), len(eval_dl)))

    return train_dl, eval_dl


def build_model(dev=None):
    """Build neural network model."""

    net = UNet(3, NUM_CLASSES)
    if GPU:
        net = nn.DataParallel(net, device_ids=DEVICE_ID)
        if dev is None:
            dev = torch.device('cuda:{}'.format(DEVICE_ID[0]))
        net.to(dev)

        if PRETRAINED:
            # 使用nn.DataParallel后保存的权重会多一层.module封装
            state_dict = torch.load(CHECKPOINT).module.state_dict()
            net.module.load_state_dict(state_dict)
            print("load pre-trained weight:{}".format(CHECKPOINT))
    else:
        if PRETRAINED:
            state_dict = torch.load(CHECKPOINT).state_dict()
            net.load_state_dict(state_dict)
            print("load pre-trained weight:{}".format(CHECKPOINT))

    # 使用同步的批次归一化
    if SYN_BN:
        print("Use Synchronized Batch Normalization.")
        pass

    return net


def set_optim(net):
    """Set optimizer and learning rate scheduler."""

    # 优化器
    optim = torch.optim.Adam(net.parameters(), BASE_LR)
    for params in optim.param_groups:
        params['lr'] = BASE_LR

    # 学习率策略
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, EPOCHS // 4, eta_min=MIN_LR)
    # 动态调整的余弦退火，初始周期为4，即４个周期后restart为初始学习率，之后呈平方增长
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, 4, T_mult=2, eta_min=MIN_LR)
    return optim, scheduler


def set_lr(optim, lr):
    for param_group in optim.param_groups:
        param_group['lr'] = lr


def train_one_epoch(epoch, train_dataloader, net, optim, loss_func, dev):
    # 一个训练周期的平均loss
    total_loss = 0.
    # 记录训练用时
    start_time = time.time()

    # 进度条
    progress = tqdm(train_dataloader)
    progress.set_description_str("Train Epoch[{}]".format(epoch + 1))
    for it, batch_data in enumerate(progress):
        batch_images = batch_data.get('image').to(dev).float()
        batch_labels = batch_data.get('label').to(dev).int()

        # 清空累积梯度
        optim.zero_grad()
        # 前向反馈
        outputs = net(batch_images)
        # 计算loss
        loss = loss_func(outputs, batch_labels)
        # 反向传播梯度
        loss.backward()
        # 优化器更新参数
        optim.step()

        total_loss += loss.detach().item()

        if it % LOG_CYCLE == 0:
            batch_image_names = batch_data.get('image_name')
            progress.set_postfix_str("Iter[{}]: loss={:.3f}, images:{}".format(it + 1, loss.item(), batch_image_names))

    end_time = time.time()
    total_loss /= len(train_dataloader)
    print("Epoch[{}]: loss={}, time used:{:.3f}s".format(epoch + 1, total_loss, end_time - start_time))
    print('-' * 60, '\n')

    return total_loss


def eval(epoch, eval_dataloader, net, criteria_func, dev):
    # 平均loss
    total_loss = 0.
    # 平均dice
    total_metric = 0.
    # 阳性样本数量
    total_pos_num = 0
    # 记录评估用时
    start_time = time.time()

    progress = tqdm(eval_dataloader)
    progress.set_description_str("Eval Epoch[{}]".format(epoch + 1))
    for it, batch_data in enumerate(progress):
        batch_labels = batch_data.get('label').to(dev).int()
        # 若该批次没有阳性样本，则略过
        if torch.sum(batch_labels > 0) == 0:
            continue

        # 该批次的阳性样本数量
        batch_pos_num = 0
        # 该批次阳性样本的平均loss和平均dice
        batch_loss = batch_metric = 0.
        batch_images = batch_data.get('image').to(dev).float()
        for image, label in zip(batch_images, batch_labels):
            # 仅对阳性样本评估
            if torch.sum(label == 1) > 0:
                # (C,H,W)->(1,C,H,W)
                image = image.unsqueeze(0)
                # (H,W)->(1,H,W)
                label = label.unsqueeze(0)
                output = net(image)
                loss, cost = criteria_func(output, label, eval=True)
                batch_pos_num += 1
                batch_loss += loss
                batch_metric += cost

        if it % LOG_CYCLE == 0:
            batch_image_names = batch_data.get('image_name')
            progress.set_postfix_str("Iter[{}]: loss={:.3f}, cost={:.3f}, images:{}".format(
                it + 1, batch_loss / batch_pos_num, batch_metric / batch_pos_num, batch_image_names))

        total_loss += batch_loss
        total_metric += batch_metric
        total_pos_num += batch_pos_num

    end_time = time.time()
    # 所有阳性样本的平均loss和dice
    total_loss /= total_pos_num
    total_metric /= total_pos_num
    print("Eval Epoch[{}]: loss={:.3f}, metric={:.3f}, time used:{:.3f}s".format(
        epoch + 1, total_loss, total_metric, end_time - start_time))
    print('-' * 60, '\n')

    return total_loss, total_metric


if __name__ == '__main__':
    if GPU:
        print("use gpu with devices:{}".format(DEVICE_ID))
        device = torch.device('cuda:{}'.format(DEVICE_ID[0]))
    else:
        device = torch.device('cpu')
    print("Train with main device [{}]\n".format(device))

    # 数据加载
    if not TINY_TRAIN:
        train_dl, eval_dl = load_data()
    else:
        # 少样本训练，以验证方法是否可靠
        train_dl = draw_tiny_samples([1.] * 1000 + [.7] * 732, TINY_NUM, TINY_BATCH_SIZE)
        eval_dl = None

    # 模型构建
    model = build_model(dev=device)
    # 定制优化器和学习率策略
    optimizer, scheduler = set_optim(model)
    # 损失函数
    criterion = Dice()
    # 可视化
    train_wtr = SummaryWriter(os.path.join(VISUAL_DIR, 'Train'))
    eval_wtr = SummaryWriter(os.path.join(VISUAL_DIR, 'Eval'))

    print("Start Training!")
    prev_metric = sys.maxsize
    for epoch in range(EPOCHS):
        # 设置模型为训练模式
        model.train()

        # 先使用小的学习率进行热身
        if not TINY_TRAIN and epoch < WARM_UP_EPOCH:
            set_lr(optimizer, WARM_UP_LR)
        # 预热一定周期后恢复初始学习率
        elif epoch == WARM_UP_EPOCH:
            set_lr(optimizer, BASE_LR)
        lr = optimizer.param_groups[0]['lr']
        # 可视化学习率曲线
        train_wtr.add_scalar('lr', lr, epoch)
        print("Epoch[{}] lr={}".format(epoch + 1, lr))

        # 训练一个周期得到平均损失
        epoch_loss = train_one_epoch(epoch, train_dl, model, optimizer, criterion, device)
        # 可视化loss曲线
        train_wtr.add_scalar('loss', epoch_loss, epoch)

        # 训练一定周期后在在验证集上测试
        if eval_dl is not None and (epoch + 1) % TIME_TO_EVAL == 0:
            print("Start Evaluation of Epoch[{}]!".format(epoch + 1))
            # 设置模型为评估模式
            model.eval()
            loss, metric = eval(epoch, eval_dl, model, criterion, device)
            eval_wtr.add_scalar('loss', loss, epoch)
            eval_wtr.add_scalar('dice', metric, epoch)

            # 若当前评估性能优于之前，则保存权重
            if metric < prev_metric:
                print("Gain best dice:{:.3f}".format(metric))
                f = os.path.join(CHECKPOINT, 'best.pt')
                torch.save(model, f)
                print("saved weights to {}\n".format(f))
                prev_metric = metric

        # 预热期过后，按照策略更新学习率
        if not TINY_TRAIN and epoch >= WARM_UP_EPOCH:
            scheduler.step()

    train_wtr.close()
    eval_wtr.close()
