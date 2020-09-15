"""数据预处理和分析相关"""

import os
import cv2
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
# PIL默认会检查解压炸弹DOS攻击，由于某些阳性样本尺寸太大，因此这里忽略检查
Image.MAX_IMAGE_PIXELS = None


def extract_files(src_dir, dst_dir):
    file_names = os.listdir(src_dir)

    for file_name in file_names:
        dst_file_name = file_name.strip()
        src_file_path = os.path.join(src_dir, file_name)

        if 'mask' in file_name:
            dst_label_dir = os.path.join(os.path.dirname(dst_dir), 'Labels')
            dst_file_path = os.path.join(dst_label_dir, dst_file_name)
        else:
            dst_file_path = os.path.join(dst_dir, dst_file_name)

        shutil.copy(src_file_path, dst_file_path)
        print("file {} done!".format(file_name))


def show_image(img, title='image', width=700, height=817):
    cv2.namedWindow(title, 0)
    cv2.resizeWindow(title, width, height)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cal_sizes(root, image_files, train=True):
    # 所有图像文件的路径
    image_paths = [os.path.join(root, image_file) for image_file in image_files]
    print("Total {} images".format(len(image_paths)))
    # （宽，高）
    sizes = np.array([Image.open(image_path).size for image_path in image_paths]).astype('float')
    print(sizes.shape)
    Ws, Hs = np.hsplit(sizes, 2)
    print(Ws.shape, Hs.shape)
    mean_w = np.mean(Ws)
    mean_h = np.mean(Hs)
    print("mean width:{}; mean height:{}".format(mean_w, mean_h))
    print("mean_w/mean_h={}".format(mean_w / mean_h))

    # 针对训练集的统计，测试集全是阳性样本
    if train:
        # 分别统计下阳性样本和阴性样本的尺寸均值
        # 先统计阳性样本
        pos_path = [os.path.join(root, filename)
                    for filename in image_files if int(filename.split('.')[0].strip()) >= 2000]
        print("Total {} positive samples".format(len(pos_path)))
        pos_sizes = np.array([Image.open(path).size for path in pos_path]).astype('float')
        pos_Ws, pos_Hs = np.hsplit(pos_sizes, 2)
        mean_pos_w = np.mean(pos_Ws)
        mean_pos_h = np.mean(pos_Hs)
        print("positive mean width:{}; positive mean height:{}".format(mean_pos_w, mean_pos_h))
        print("pos_w/pos_h={}".format(mean_pos_w / mean_pos_h))

        # 先统计阴性样本
        neg_path = [path for path in image_paths if path not in pos_path]
        print("Total {} negative samples".format(len(neg_path)))
        neg_sizes = np.array([Image.open(path).size for path in neg_path]).astype('float')
        neg_Ws, neg_Hs = np.hsplit(neg_sizes, 2)
        mean_neg_w = np.mean(neg_Ws)
        mean_neg_h = np.mean(neg_Hs)
        print("negative mean width:{}; negative mean height:{}".format(mean_neg_w, mean_neg_h))
        print("neg_w/neg_h={}".format(mean_neg_w / mean_neg_h))


def cal_hist(image, channel=0):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist(image, [channel], None, [256], [0, 256])
    max_num = hist.max()
    max_num_pixel = hist.argmax()
    print("num:{}; pixel:{}; ratio:{}".format(max_num, max_num_pixel, max_num / hist.sum()))

    plt.plot(hist)
    plt.title('hist')
    plt.show()

    return hist


def _mean_std_calc(paths, channel):
    channel_map = {0: 'B', 1: 'G', 2: 'R'}

    # 像素数量
    num = 0
    # 像素值总和
    value_sum = 0
    # 像素值平方的总和
    square_sum = 0

    for path in paths:
        # BGR
        img = cv2.imread(path)
        h, w, _ = img.shape
        num += h * w

        img_c = img[:, :, channel]
        value_sum += np.sum(img_c)
        square_sum += np.sum(np.power(img_c, 2.))

    mean = value_sum / num
    print("@mean of channel[{}]: {}".format(channel_map[channel], mean))
    std = np.sqrt(square_sum / num - mean ** 2)
    print("@std of channel[{}]: {}".format(channel_map[channel], std))

    return mean, std


def pixel_calc(image_paths, channel, train=True):
    print("Total {} samples".format(len(image_paths)))
    # 统计所有样本的单通道像素均值与方差
    print("Start to calculate all samples..")
    mean, std = _mean_std_calc(image_paths, channel)
    print("Done!")

    if train:
        # 阳性样本图像路径
        pos_paths = [path for path in image_paths if int(os.path.basename(path).split('.')[0]) > 1000]
        print("Total {} positive samples".format(len(pos_paths)))
        # 统计阳性样本的单通道像素均值与方差
        print("Start to calculate positive samples..")
        pos_mean, pos_std = _mean_std_calc(pos_paths, channel)
        print("Done!")

        # 阴性样本图像路径
        neg_paths = [path for path in image_paths if int(os.path.basename(path).split('.')[0]) < 1000]
        print("Total {} negative samples".format(len(neg_paths)))
        # 统计阴性样本的单通道像素均值与方差
        print("Start to calculate negative samples..")
        neg_mean, neg_std = _mean_std_calc(neg_paths, channel)
        print("Done!")

        return mean, std, pos_mean, pos_std, neg_mean, neg_std
    else:
        return mean, std


def calc_train_eval_pixel(train_img_paths, eval_img_paths):
    # 统计训练集与验证集的像素均值、标准差，注意是BGR通道顺序
    train_mean, train_std = [], []
    eval_mean, eval_std = [], []
    for c in range(3):
        train_mean_c, train_std_c = pixel_calc(train_img_paths, c, train=False)
        eval_mean_c, eval_std_c = pixel_calc(eval_img_paths, c, train=False)

        train_mean.append(train_mean_c)
        train_std.append(train_std_c)
        eval_mean.append(eval_mean_c)
        eval_std.append(eval_std_c)

    c_map = {0: 'R', 1: 'G', 2: 'B'}

    train_mean = train_mean[::-1]
    train_std = train_std[::-1]
    print('[Train]')
    for c, (mean, std) in enumerate(zip(train_mean, train_std)):
        print("#{}: mean={:.3f}, std={:.3f}".format(c_map[c], mean, std))

    eval_mean = eval_mean[::-1]
    eval_std = eval_std[::-1]
    print('[Eval]')
    for c, (mean, std) in enumerate(zip(eval_mean, eval_std)):
        print("#{}: mean={:.3f}, std={:.3f}".format(c_map[c], mean, std))


def calc_scale_pixel(ds, size, dev):
    """统计数据集缩放后的图像像素均值"""

    # 像素数量
    num = size[0] * size[1] * len(ds)
    # 各通道像素值的和
    value_sum_r = value_sum_g = value_sum_b = 0
    # 各通道像素值平方的和
    square_sum_r = square_sum_g = square_sum_b = 0

    for data in ds:
        # (C,H,W), float32
        image_ts = data.get('image').to(dev)

        value_sum_r += image_ts[0].sum().item()
        value_sum_g += image_ts[1].sum().item()
        value_sum_b += image_ts[2].sum().item()

        square_sum_r += image_ts[0].pow(2.).sum().item()
        square_sum_g += image_ts[1].pow(2.).sum().item()
        square_sum_b += image_ts[2].pow(2.).sum().item()

    scale_mean_r = value_sum_r / num
    scale_mean_g = value_sum_g / num
    scale_mean_b = value_sum_b / num
    print("scale mean r={:.3f}, g={:.3f}, b={:.3f}".format(scale_mean_r, scale_mean_g, scale_mean_b))

    scale_std_r = np.sqrt(square_sum_r / num - scale_mean_r ** 2)
    scale_std_g = np.sqrt(square_sum_g / num - scale_mean_g ** 2)
    scale_std_b = np.sqrt(square_sum_b / num - scale_mean_b ** 2)
    print("scale std r={:.3f}, g={:.3f}, b={:.3f}".format(scale_std_r, scale_std_g, scale_std_b))


def mask_analysis(label_dir):
    label_files = os.listdir(label_dir)
    print("Total {} labels".format(len(label_files)))

    label_paths = [os.path.join(label_dir, file) for file in label_files]
    sample_path = random.choice(label_paths)
    print("pick label sample {}".format(os.path.basename(sample_path)))

    # 用opencv读默认是3通道
    sample_label = cv2.imread(sample_path)
    h, w, c = sample_label.shape
    # uint8类型
    print(sample_label.dtype)
    # 3通道
    print("height:{}; width:{}; channel:{}".format(h, w, c))

    # mask并非二值的，发现有20种像素值，0-9，247-255
    print(np.unique(sample_label))
    show_image(sample_label, title='label', width=w, height=h)

    # PIL读进来发现其实mask是单通道
    sample_mask = Image.open(sample_path)
    print(sample_mask)
    # L
    print(sample_mask.mode)
    mask = np.asarray(sample_mask)
    # 转换成numpy矩阵后是二维的，uint8类型
    print(mask.shape, mask.dtype)


def result_analysis(result_dir):
    result_files = os.listdir(result_dir)
    result_path = [os.path.join(result_dir, file) for file in result_files]
    result_images = [cv2.imread(path) for path in result_path]

    for result in result_images:
        # 发现仅有0和255两种像素值
        # 因此只需要做二分类即可
        print(np.unique(result))
        # uint8类型
        print(result.dtype)
        # 3通道
        print(result.shape)


def gen_mask_for_negatives(label_dir, paths):
    for path in paths:
        neg = Image.open(path)
        w, h = neg.size
        print("width:{}; height:{}".format(w, h))

        filename = os.path.basename(path)
        label_name = filename.split('.')[0].strip() + '_mask.' + filename.split('.')[1].strip()
        print(label_name)
        label_path = os.path.join(label_dir, label_name)

        # 若是dtype='int'，则会变成int32类型，注意！
        mask = np.zeros((h, w), dtype=np.uint8)
        assert mask.ndim == 2 and mask.dtype == np.uint8
        label = Image.fromarray(mask)
        print(label.mode)
        label.save(label_path)


if __name__ == '__main__':
    # 将图像提取到对应目录
    # src = ['data01', 'data02', 'data03']
    # dst = 'Data/Train/Images'
    #
    # for s in src:
    #     extract_files(s, dst)
    # src = 'Data/Train/data04'
    # dst = 'Data/Train/Images'
    # extract_files(src, dst)

    # 训练集目录
    # root = 'Data/Train/Images'
    # 测试集目录
    root = 'Data/Test/Images'
    # 所有图像文件名称
    image_files = os.listdir(root)
    # # 随机挑选一张样本做分析
    # sample_file = random.choice(image_files)
    # print("pick sample:{}".format(sample_file))
    # sample_path = os.path.join(root, sample_file)
    # sample = Image.open(sample_path)
    # W, H = sample.size
    # # RGB
    # mode = sample.mode
    # print("width:{}; height:{}; mode:{}".format(W, H, mode))
    # # uint8类型
    # sample_arr = np.asarray(sample)
    # print(sample_arr.dtype)
    # sample.show()

    # image = cv2.cvtColor(np.asarray(sample), cv2.COLOR_RGB2BGR)
    # H, W, C = image.shape
    # show_image(image, title='sample', width=W, height=H)

    # 统计训练集的图像尺寸
    # cal_sizes(root, image_files)
    # 统计测试集的图像尺寸
    # cal_sizes(root, image_files, train=False)

    # 像素直方图统计
    # hist = cal_hist(image)

    # copy = np.zeros_like(image)
    #     # pos = np.bitwise_and(image > 230, image < 250)
    #     # copy[pos] = 255
    #     # show_image(copy, title='copy')

    # # 分析阴性样本的像素集中区域
    # neg_file = random.choice(image_files)
    # while int(neg_file.split('.')[0]) > 1000:
    #     neg_file = random.choice(image_files)
    #
    # print("pick negative sample:{}".format(neg_file))
    # neg_path = os.path.join(root, neg_file)
    # neg_sample = cv2.imread(neg_path)
    # # 统计像素直方图
    # hist = cal_hist(neg_sample)
    # # print(hist)
    # pixels = np.argsort(hist.squeeze())[::-1]
    # # 输出top10像素值
    # print(pixels[:10])
    # # 输出像素均值和方差
    # print("mean:{}; std:{}".format(neg_sample.mean(), neg_sample.std()))
    #
    # # 分析阳性样本的像素集中区域
    # pos_file = random.choice(image_files)
    # while int(pos_file.split('.')[0]) < 2000:
    #     pos_file = random.choice(image_files)
    #
    # print("pick positive sample:{}".format(pos_file))
    # pos_path = os.path.join(root, pos_file)
    # pos_sample = cv2.imread(pos_path)
    # # 统计像素直方图
    # hist = cal_hist(pos_sample)
    # # print(hist)
    # pixels = np.argsort(hist.squeeze())[::-1]
    # # 输出top10像素值
    # print(pixels[:10])
    # # 输出像素均值和方差
    # print("mean:{}; std:{}".format(pos_sample.mean(), pos_sample.std()))

    # 统计样本的像素均值与标准差
    # 所有图像文件的路径
    image_paths = [os.path.join(root, image_file) for image_file in image_files]
    for c in range(3):
        pixel_calc(image_paths, c, train=False)

    # 为阴性样本生成标注mask
    # label_dir = 'Data/Train/Labels'
    # neg_paths = [path for path in image_paths if int(os.path.basename(path).split('.')[0]) < 1000]
    # assert len(neg_paths) == 1000
    # gen_mask_for_negatives(label_dir, neg_paths)

    # 标注mask分析
    # label_dir = 'Data/Train/Labels'
    # mask_analysis(label_dir)

    # 提交示例图像分析
    # result_dir = 'Data/Result/sample'
    # result_analysis(result_dir)
