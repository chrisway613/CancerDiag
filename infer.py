import os
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from datetime import datetime

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import Compose

from conf import *
from models import *
from data_process import calc_scale_pixel

from Data.load_data import CancerDataset
from Data.transform import ConvertToTensor, Scale, Norm


if __name__ == '__main__':
    # 存放预测结果的目录
    today = datetime.now().strftime('%Y-%m-%d')
    result_dir = os.path.join(RESULT_DIR, today)
    os.makedirs(result_dir, exist_ok=True)

    if GPU:
        print("use gpu with devices:{}".format(DEVICE_ID))
        device = torch.device('cuda:{}'.format(DEVICE_ID[0]))
    else:
        device = torch.device('cpu')
    print("Infer with main device [{}]\n".format(device))

    # 加载测试集
    path = os.path.join(DATA_DIR, 'Test')
    test_ds = CancerDataset(root=path, train=False, transform=Compose([
        ConvertToTensor(),
        Scale(size=(INPUT_SIZE[1], INPUT_SIZE[0]), mode='bilinear', align_corners=True),
        Norm(mean=TEST_SCALE_MEAN, std=TEST_SCALE_STD)
    ]))
    # calc_scale_pixel(test_ds, INPUT_SIZE, device)
    # batch size=1
    test_dl = DataLoader(test_ds, num_workers=8, pin_memory=True)
    print("Total {} batches of {} test images\n".format(len(test_dl), len(test_ds)))

    # 加载模型
    model = UNet(3, NUM_CLASSES)
    if GPU:
        model = nn.DataParallel(model, device_ids=DEVICE_ID)
        model.to(device)
        state_dict = torch.load(CHECKPOINT).module.state_dict()
        model.module.load_state_dict(state_dict)
    else:
        state_dict = torch.load(CHECKPOINT).state_dict()
        model.load_state_dict(state_dict)
    print("Load weights:{}\n".format(CHECKPOINT))

    wtr = SummaryWriter(os.path.join(VISUAL_DIR, 'Test'))

    # 对测试集进行预测
    with torch.no_grad():
        print("Start testing!")
        start_time = time.time()

        for num, data in enumerate(test_dl):
            # (1,3,h,w)
            image = data.get('image').float().to(device)
            image_name = data.get('image_name')[0]
            image_width, image_height = data.get('image_size')
            print("[{}] predicting..".format(image_name))

            # (1,1,h,w)
            output = model(image)
            # (1,1,h,w)，预测概率图
            prob = torch.sigmoid(output)
            # 基于阀值生成预测二值图，即mask
            pred = torch.zeros_like(prob, device=device)
            pred[prob >= THRESH] = 1

            # 可视化预测结果
            # (1,3,h,w)->(h,w,3)
            image = image.cpu().squeeze().permute(1, 2, 0)
            # 反归一化
            image = image.mul_(
                torch.as_tensor(TEST_SCALE_STD, dtype=image.dtype)
            ).add_(torch.as_tensor(TEST_SCALE_MEAN, dtype=image.dtype))
            # (h,w,3) -> (3,h,w)
            image = image.permute(2, 0, 1)
            wtr.add_image('image-{}'.format(image_name), image, num)
            # (2,1,h,w)
            concat = torch.cat([prob, pred])
            mask_grids = make_grid(concat, nrow=2, padding=3, pad_value=.5)
            wtr.add_image('mask-{}'.format(image_name), mask_grids, num)

            # 将预测mask插值到原图尺寸 (高,宽)
            image_size = (image_height.item(), image_width.item())
            # mask使用最近邻插值 (1,1,h,w)->(1,1,H,W)
            pred = F.interpolate(pred, size=image_size)
            # 解码：0、1 -> 0、255
            pred *= 255

            # 生成单通道uint8类型的mask
            # (H,W)
            pred_arr = pred.squeeze().cpu().numpy().astype('uint8')
            assert len(np.unique(pred_arr)) <= 2
            mask = Image.fromarray(pred_arr)
            # (宽,高)
            assert mask.size == (image_size[1], image_size[0]) and mask.mode == 'L'

            # 保存结果到图像文件
            name = image_name.split('.')[0].strip() + '_mask.jpg'
            mask.save(os.path.join(result_dir, name))

            print("[{}] done!\n".format(image_name))

        print("Finished, time used:{:.3f}s".format(time.time() - start_time))
