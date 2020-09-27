import os
import time
import shelve
import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import Compose

from conf import *
from utils import *
from models import *
from SynBn import convert_model
from Data.misc import train_eval_split, draw_tiny_samples, load_balanced_data, get_mask_pos_rate
from data_process import calc_train_eval_pixel, calc_scale_pixel

from Data.load_data import CancerDataset
from Data.transform import ConvertToTensor, Norm, Scale, RandomFlip, SomeAugs

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(device_id) for device_id in DEVICE_ID])


def load_data():
    """Build Dataloader of training & validation set."""
    #
    # if GPU:
    #     dev = torch.device('cuda:{}'.format(DEVICE_ID[0]))
    # else:
    #     dev = torch.device('cpu')

    # 从总体训练集中划分一部分作为验证集
    try:
        # 从文件读取之前已经划分好的结果
        with shelve.open(DS_PATHS, 'r') as db:
            train_img_paths = db['train_img_paths']
            train_label_paths = db['train_label_paths']
            # 阳性样本路径，此处代表仅训练阳性样本
            # train_img_paths = db['train_img_pos_paths']
            # train_label_paths = db['train_label_pos_paths']
            eval_img_paths = db['eval_img_paths']
            eval_label_paths = db['eval_label_paths']
    except:
        # 若文件不存在则重新划分
        train_img_paths, eval_img_paths, train_label_paths, eval_label_paths = train_eval_split()
    # # 统计划分后在原图尺寸下训练集和验证集的像素均值
    # calc_train_eval_pixel(train_img_paths, eval_img_paths)

    # # 构造训练集
    # train_ds = CancerDataset(
    #     img_paths=train_img_paths,
    #     label_paths=train_label_paths,
    #     transform=Compose([
    #         # RandomFlip(prob_h=.3, prob_v=.1),
    #         # SomeAugs(),
    #         # PILResize(INPUT_SIZE, mode=Image.BILINEAR),
    #         # 转换为0-1张量
    #         ConvertToTensor(),
    #         # 尺寸缩放，短边（高）缩放到512，同时保持基于训练集图像统计的宽高比均值
    #         # Scale(size=(INPUT_SIZE[1], INPUT_SIZE[0]), mode='bicubic', align_corners=True),
    #         Scale(size=(INPUT_SIZE[1], INPUT_SIZE[0]), mode='bilinear', align_corners=True),
    #         # 归一化
    #         Norm(mean=TRAIN_SCALE_MEAN, std=TRAIN_SCALE_STD)
    #         # Norm(mean=TRAIN_SCALE_POS_MEAN, std=TRAIN_SCALE_POS_STD)
    #     ]),
    #     # 如果排序，Dataloader drop_last=True时会将最后几个阳性样本drop掉，而我们构造训练集样本路径时是将阳性样本排前面的
    #     sort=False
    # )
    # # 统计缩放后训练集图像的均值和标准差
    # calc_scale_pixel(train_ds, INPUT_SIZE, dev)
    # # 统计数据集mask的病灶面积占比
    # get_mask_pos_rate(train_ds, INPUT_SIZE)

    # # 构造验证集
    # eval_ds = CancerDataset(
    #     img_paths=eval_img_paths,
    #     label_paths=eval_label_paths,
    #     transform=Compose([
    #         # PILResize(INPUT_SIZE, mode=Image.BILINEAR),
    #         # 转换为0-1张量
    #         ConvertToTensor(),
    #         # 尺寸缩放，短边（高）缩放到512，同时保持基于训练集图像统计的宽高比均值
    #         # Scale(size=(INPUT_SIZE[1], INPUT_SIZE[0]), mode='bicubic', align_corners=True),
    #         Scale(size=(INPUT_SIZE[1], INPUT_SIZE[0]), mode='bilinear', align_corners=True),
    #         # 归一化
    #         Norm(mean=EVAL_SCALE_MEAN, std=EVAL_SCALE_STD)
    #     ])
    # )
    # # 统计缩放后验证集图像的均值和标准差
    # calc_scale_pixel(eval_ds, INPUT_SIZE, dev)

    # assert train_ds[0].get('image').shape[1:] == (INPUT_SIZE[1], INPUT_SIZE[0])
    # assert eval_ds[1].get('image').shape[1:] == (INPUT_SIZE[1], INPUT_SIZE[0])
    # print("Train with image size(width, height): {}".format(INPUT_SIZE))

    # # num_workers设置为内核数量能加快加载batch的耗时，节省一倍多训练时间；pin_memory=True也是，与数据加载时内存锁页相关
    # train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    # eval_dl = DataLoader(eval_ds, BATCH_SIZE, num_workers=8, pin_memory=True)

    train_ds, eval_ds = load_balanced_data(train_img_paths, eval_img_paths, train_label_paths, eval_label_paths)
    # # 统计数据集mask的病灶面积占比
    # get_mask_pos_rate(train_ds, INPUT_SIZE)
    # # 统计像素均值和标准差
    # calc_scale_pixel(train_ds, INPUT_SIZE, dev)
    # calc_scale_pixel(eval_ds, INPUT_SIZE, dev)

    # num_workers设置为内核数量能加快加载batch的耗时，节省一倍多训练时间；pin_memory=True也是，与数据加载时内存锁页相关
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    eval_dl = DataLoader(eval_ds, BATCH_SIZE, num_workers=8, pin_memory=True)
    print("batch size={}, {} batches in training set, {} batches in validation set\n".format(
        BATCH_SIZE, len(train_dl), len(eval_dl)))

    # # 验证每个batch都有阴阳样本(this may take few minutes)
    # for batch_data in train_dl:
    #     batch_labels = batch_data.get('label')
    #     batch_label_names = batch_data.get('label_name')
    #     assert torch.sum(batch_labels > 0) > 0, "there is no positive sample in {}!".format(
    #         batch_label_names.numpy().tolist())
    # for batch_data in eval_dl:
    #     batch_labels = batch_data.get('label')
    #     batch_label_names = batch_data.get('label_name')
    #     assert torch.sum(batch_labels > 0) > 0, "there is no positive sample in {}!".format(
    #         batch_label_names.numpy().tolist())

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
            state_dict = torch.load(WEIGHT).module.state_dict()
            net.module.load_state_dict(state_dict)
            print("load pre-trained weight: {}".format(WEIGHT))
    else:
        if PRETRAINED:
            state_dict = torch.load(WEIGHT).state_dict()
            net.load_state_dict(state_dict)
            print("load pre-trained weight: {}".format(WEIGHT))

    # 使用同步的批次归一化
    if SYN_BN:
        net = convert_model(net)
        print("Use Synchronized Batch Normalization\n")

    return net


def set_optim(net):
    """Set optimizer and learning rate scheduler."""

    # 优化器
    optim = torch.optim.Adam(net.parameters(), BASE_LR)
    for params in optim.param_groups:
        params['lr'] = BASE_LR

    # 学习率策略
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 5, eta_min=MIN_LR)
    # 动态调整的余弦退火，初始周期为4，即4个周期后restart为初始学习率，之后呈平方增长
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, 4, T_mult=2, eta_min=MIN_LR)
    return optim, scheduler


def set_lr(optim, lr):
    for param_group in optim.param_groups:
        param_group['lr'] = lr


def train_one_epoch(epoch, train_dataloader, net, optim, loss_func, dev, wtr):
    # 一个训练周期的平均loss
    total_loss = 0.
    # # 加上正则化的loss
    # total_reg_loss = 0.
    total_bce_loss = 0.
    total_dice_loss = 0.
    # 记录训练用时
    start_time = time.time()

    # 进度条
    progress = tqdm(train_dataloader)
    progress.set_description_str("Train Epoch[{}]".format(epoch + 1))
    for it, batch_data in enumerate(progress):
        # (N,C,H,W)
        batch_images = batch_data.get('image').to(dev).float()
        # (N,H,W)
        batch_labels = batch_data.get('label').to(dev).int()

        # 清空累积梯度
        optim.zero_grad()
        # 前向反馈
        outputs = net(batch_images)
        # 计算loss
        # loss, _ = loss_func(outputs, batch_labels)
        loss, bce_loss, dice_loss, bce_indices, dice_indices = loss_func(outputs, batch_labels)

        # # L2正则化
        # l2 = 0.
        # for name, param in model.named_parameters():
        #     # 仅对权重施加(忽略bias)
        #     if 'weight' in name:
        #         # L2范数的平方
        #         l2 += torch.norm(param) ** 2
        #
        # # WEIGHT_DECAY - 正则化惩罚系数
        # reg_loss = loss + WEIGHT_DECAY / (2 * train_dataloader.batch_size) * l2

        # 反向传播梯度
        loss.backward()
        # reg_loss.backward()
        # 优化器更新参数
        optim.step()

        total_bce_loss += bce_loss.item()
        total_dice_loss += dice_loss.item()
        total_loss += loss.detach().item()
        # total_reg_loss += reg_loss.detach().item()

        if it % LOG_CYCLE == 0:
            batch_image_names = batch_data.get('image_name')
            batch_image_names_arr = np.asarray(batch_image_names)
            bce_top_k_images = batch_image_names_arr[bce_indices.cpu().numpy()]
            dice_top_k_images = batch_image_names_arr[dice_indices.cpu().numpy()]
            # progress.set_postfix_str("Iter[{}]: loss={:.5f}, images:{}".format(it + 1, loss.item(), batch_image_names))
            progress.set_postfix_str("Iter[{}]: bce loss={:.5f}, dice loss={:.5f}, total loss={:.5f}".format(
                it + 1, bce_loss, dice_loss, loss))
            # progress.set_postfix_str(
            #     "Iter[{}]: bce loss={:.5f}, dice loss={:.5f}, total loss={:.5f}, reg loss={:.5f}, l2={:.3f}".format(
            #         it + 1, bce_loss, dice_loss, loss, reg_loss, l2)
            # )
            print("\nTrain Epoch[{}] Iter[{}] images: {}".format(epoch + 1, it + 1, batch_image_names))
            print("Top k images of BCE Loss: {}".format(bce_top_k_images))
            print("Top k images of Dice Loss: {}\n".format(dice_top_k_images))

        # 当前迭代次数＝周期x批次总数+当前批次
        step = epoch * len(train_dataloader) + it
        if step % VIS_CYCLE == 0:
            batch_image_names = batch_data.get('image_name')
            for output, label, name in zip(outputs, batch_labels, batch_image_names):
                # 概率图, (1,H,W)->(1,1,H,W)
                pred = torch.sigmoid(output).unsqueeze(0)
                # 二值图，非0即1
                binary_pred = torch.zeros_like(pred)
                # 概率超过预测阀值的认为是阳性区域
                binary_pred[pred > THRESH] = 1

                # 将预测结果和标注mask拼接在一起：(3,1,H,W)
                # 注意标注mask要先从(H,W)变为(1,1,H,W)并且转换为float32类型
                concat = torch.cat([pred, label.unsqueeze(0).unsqueeze(0).float(), binary_pred])
                # padding是代表图像之间的间隔距离，.5代表使用灰色作为分隔颜色
                image_grids = make_grid(concat, nrow=3, padding=3, pad_value=.5)
                wtr.add_image('Train-Epoch{}-Step{}-{}'.format(epoch + 1, it + 1, str(name)), image_grids, step)

    end_time = time.time()
    total_loss /= len(train_dataloader)
    # total_reg_loss /= len(train_dataloader)
    total_bce_loss /= len(train_dataloader)
    total_dice_loss /= len(train_dataloader)
    # print("Epoch[{}]: loss={}, time used:{:.3f}s".format(epoch + 1, total_loss, end_time - start_time))
    print("Epoch[{}]: bce loss={:.5f}, dice loss={:.5f}, loss={:.5f}, time used:{:.3f}s".format(
        epoch + 1, total_bce_loss, total_dice_loss, total_loss, end_time - start_time))
    # print("Epoch[{}]: bce loss={:.5f}, dice loss={:.5f}, loss={:.5f}, reg loss={:.5f}, time used:{:.3f}s".format(
    #     epoch + 1, total_bce_loss, total_dice_loss, total_loss, total_reg_loss, end_time - start_time))
    print('-' * 60, '\n')

    # 释放缓存的GPU资源
    torch.cuda.empty_cache()

    # return total_loss
    return total_loss, total_bce_loss, total_dice_loss
    # return total_reg_loss, total_loss, total_bce_loss, total_dice_loss


def eval(epoch, eval_dataloader, net, criteria_func, dev, wtr):
    # 平均loss
    total_loss = 0.
    total_bce_loss = 0.
    total_dice_loss = 0.
    # 平均dice
    total_dice = 0.
    # 阳性样本数量
    total_pos_num = 0
    # 记录评估用时
    start_time = time.time()

    # 加上这句可减少GPU占用
    with torch.no_grad():
        # 使用dice评估
        metric_func = Dice()

        # 进度条
        progress = tqdm(eval_dataloader)
        progress.set_description_str("Eval Epoch[{}]".format(epoch + 1))
        for it, batch_data in enumerate(progress):
            batch_labels = batch_data.get('label').int()
            # 若该批次没有阳性样本，则略过
            if torch.sum(batch_labels > 0) == 0:
                continue

            batch_image_names = batch_data.get('image_name')
            batch_images = batch_data.get('image').float()

            # 该批次的阳性样本数量
            batch_pos_num = 0
            # 该批次阳性样本的平均loss和平均dice
            batch_loss = batch_dice = 0.
            batch_dice_loss = batch_bce_loss = 0.

            for image, label, name in zip(batch_images, batch_labels, batch_image_names):
                # 仅对阳性样本评估
                if torch.sum(label == 1) > 0:
                    # (C,H,W)->(1,C,H,W)
                    image = image.unsqueeze(0).to(dev)
                    # (H,W)->(1,H,W)
                    label = label.unsqueeze(0).to(dev)
                    # (1,1,H,W)
                    output = net(image)
                    # loss, _ = criteria_func(output, label)
                    loss, bce_loss, dice_loss, _, _ = criteria_func(output, label)
                    dice = metric_func.get_dice(output, label)
                    batch_pos_num += 1
                    batch_bce_loss += bce_loss.item()
                    batch_dice_loss += dice_loss.item()
                    batch_loss += loss.detach().item()
                    batch_dice += dice

                    step = epoch * len(eval_dataloader) + it
                    if step % VIS_CYCLE == 0:
                        # 概率图, (1,1,H,W)
                        pred = torch.sigmoid(output)
                        # 二值图
                        binary_pred = torch.zeros_like(pred)
                        binary_pred[pred > THRESH] = 1
                        # (1,H,W)->(1,1,H,W)
                        mask = label.unsqueeze(0).float()

                        concat = torch.cat([pred, mask, binary_pred])
                        image_grids = make_grid(concat, nrow=3, padding=3, pad_value=.5)
                        wtr.add_image('Eval-Epoch{}-Step{}-{}'.format(epoch + 1, it + 1, name), image_grids, step)

            if it % LOG_CYCLE == 0:
                batch_image_names = batch_data.get('image_name')
                # progress.set_postfix_str("Iter[{}]: loss={:.5f}, dice={:.5f}, images:{}".format(
                #     it + 1, batch_loss / batch_pos_num, batch_dice / batch_pos_num, batch_image_names))
                progress.set_postfix_str("Iter[{}]: bce loss={:.5f}, dice loss={:.5f}, loss={:.5f}, "
                                         "dice={:.5f}".format(it + 1, batch_bce_loss / batch_pos_num,
                                                              batch_dice_loss / batch_pos_num,
                                                              batch_loss / batch_pos_num,
                                                              batch_dice / batch_pos_num))
                print("\nEval Epoch[{}] Iter[{}] images: {}\n".format(epoch + 1, it + 1, batch_image_names))

            total_loss += batch_loss
            total_bce_loss += batch_bce_loss
            total_dice_loss += batch_dice_loss
            total_dice += batch_dice
            total_pos_num += batch_pos_num

    end_time = time.time()
    # 所有阳性样本的平均loss和dice
    total_loss /= total_pos_num
    total_bce_loss /= total_pos_num
    total_dice_loss /= total_pos_num
    total_dice /= total_pos_num
    # print("Eval Epoch[{}]: loss={:.5f}, dice={:.5f}, time used:{:.3f}s".format(
    #     epoch + 1, total_loss, total_dice, end_time - start_time))
    print("Eval Epoch[{}]: bce loss={:.5f}, dice loss={:.5f}, loss={}, dice={:.5f}, time used:{:.3f}s".format(
        epoch + 1, total_bce_loss, total_dice_loss, total_loss, total_dice, end_time - start_time))
    print('-' * 60, '\n')

    # 释放缓存的GPU资源
    torch.cuda.empty_cache()

    # return total_loss, total_dice
    return total_loss, total_dice, total_bce_loss, total_dice_loss


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
        train_dl = draw_tiny_samples([.7] * 1000 + [1.] * 732, TINY_NUM, TINY_BATCH_SIZE)
        eval_dl = None

    # 模型构建
    model = build_model(dev=device)
    # # 打印层
    # children = model.module.named_children() if GPU else model.named_children()
    # print("Children:")
    # for name, child in children:
    #     print("#{}".format(name))
    #     print(child)
    #     print('-' * 50)
    # print('\n')
    # # 打印参数
    # params = model.module.named_parameters() if GPU else model.named_parameters()
    # print("Params:")
    # for name, param in params:
    #     print("#{}".format(name))
    #     print(param)
    #     print('-' * 50)
    # print('\n')

    # 定制优化器和学习率策略
    optimizer, scheduler = set_optim(model)

    # 损失函数
    # criterion = Dice()
    # criterion = BCEDice()
    # # 数据集mask中，0的区域约是1的区域的7.5倍，因此这里在BCE中设置类别1的loss是类别0的7.428倍
    # criterion = BCEDice(pos_weight=torch.full((NUM_CLASSES,), 7.428, device=device))
    # # 数据集mask中，平均每个阳性样本0的区域约是1的区域的4.464倍，因此这里在BCE中设置类别1的loss是类别0的4.5倍
    # criterion = BCEDice(pos_weight=torch.full((NUM_CLASSES,), 4.5, device=device))
    # # 划分训练集（阴阳比1:10）后，在mask中，0的区域约是1的区域的3.636倍，因此这里在BCE中设置类别1的loss是类别0的3.636倍
    # criterion = BCEDice(pos_weight=torch.full((NUM_CLASSES,), 3.636, device=device))
    # # OHEM, 使用前75%的困难样本进行学习
    print("#Switch to OHEM mode, topK ratio=75%\n")
    criterion = BCEDice(pos_weight=torch.full((NUM_CLASSES,), 7.428, device=device), ohem=True, top_k_ratio=.75)

    # 可视化
    train_wtr = SummaryWriter(os.path.join(VISUAL_DIR, 'Train'))
    eval_wtr = SummaryWriter(os.path.join(VISUAL_DIR, 'Eval'))

    print("Start Training!")
    prev_dice = 0.
    for epoch in range(EPOCHS):
        # 设置模型为训练模式
        model.train()

        # Gradual warm-up，初始阶段使用线性增长的小的学习率
        if epoch < WARM_UP_EPOCH:
            print("Warm-Up Stage:[{}/{}]".format(epoch + 1, WARM_UP_EPOCH))
            set_lr(optimizer, BASE_LR * ((epoch + 1) / WARM_UP_EPOCH))

        lr = optimizer.param_groups[0]['lr']
        # 可视化学习率曲线
        train_wtr.add_scalar('lr', lr, epoch)
        print("Epoch[{}] lr={}".format(epoch + 1, lr))

        # 训练一个周期得到平均损失
        # epoch_loss = train_one_epoch(epoch, train_dl, model, optimizer, criterion, device, train_wtr)
        epoch_loss, epoch_bce_loss, epoch_dice_loss = train_one_epoch(epoch, train_dl, model, optimizer, criterion,
                                                                      device, train_wtr)
        # epoch_reg_loss, epoch_loss, epoch_bce_loss, epoch_dice_loss = train_one_epoch(
        #     epoch, train_dl, model, optimizer, criterion, device, train_wtr
        # )
        # 可视化loss曲线
        train_wtr.add_scalar('loss', epoch_loss, epoch)
        # train_wtr.add_scalar('reg loss', epoch_reg_loss, epoch)
        train_wtr.add_scalar('bce loss', epoch_bce_loss, epoch)
        train_wtr.add_scalar('dice loss', epoch_dice_loss, epoch)

        # 训练一定周期后在在验证集上测试
        if eval_dl is not None and (epoch + 1) % TIME_TO_EVAL == 0:
            print("Start Evaluation of Epoch[{}]!".format(epoch + 1))
            # 设置模型为评估模式
            model.eval()
            # loss, dice = eval(epoch, eval_dl, model, criterion, device, eval_wtr)
            loss, dice, bce_loss, dice_loss = eval(epoch, eval_dl, model, criterion, device, eval_wtr)
            eval_wtr.add_scalar('loss', loss, epoch)
            eval_wtr.add_scalar('dice', dice, epoch)
            eval_wtr.add_scalar('bce loss', bce_loss, epoch)
            eval_wtr.add_scalar('dice loss', dice_loss, epoch)

            # 若当前评估性能优于之前，则保存权重
            if dice > prev_dice:
                print("Gain best dice:{:.5f}".format(dice))
                f = os.path.join(CHECKPOINT, 'epoch{}_best_{:.5f}.pt'.format(epoch + 1, dice))
                torch.save(model, f)
                print("saved weights to {}\n".format(f))
                prev_dice = dice
            # 否则每100个epoch保存下权重
            else:
                if (epoch + 1) % 100 == 0:
                    f = os.path.join(CHECKPOINT, 'epoch{}_{:.5f}.pt'.format(epoch + 1, dice))
                    torch.save(model, f)
                    print("Epoch[{}] saved weights to {}\n".format(epoch + 1, f))

        # 预热期过后，按照策略更新学习率
        if epoch >= WARM_UP_EPOCH:
            scheduler.step()

    f = os.path.join(CHECKPOINT, 'last.pt')
    torch.save(model, f)
    print("saved weights to {}\n".format(f))

    train_wtr.close()
    eval_wtr.close()
