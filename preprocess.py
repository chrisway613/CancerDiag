import os
import cv2
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


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


def cal_sizes(root):
    # 所有图像文件的路径
    image_paths = [os.path.join(root, image_file) for image_file in image_files]
    # （宽，高）
    sizes = np.array([Image.open(image_path).size for image_path in image_paths]).astype('float')
    Ws, Hs = np.hsplit(sizes, 2)
    mean_w = np.mean(Ws)
    mean_h = np.mean(Hs)
    print("mean width:{}; mean height:{}".format(mean_w, mean_h))
    print("mean_w/mean_h={}".format(mean_w / mean_h))

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


def pixel_calc(image_paths, channel):
    # 统计所有样本的单通道像素均值与方差
    mean = np.mean([cv2.imread(path)[:, :, channel] for path in image_paths])
    print(mean)
    # std = np.std([cv2.imread(path) for path in image_paths])
    # print(std)

    # TODO: 统计阴性样本的像素均值与方差

    # TODO: 统计阳性样本的像素均值与方差


def mask_analysis(label_dir):
    label_files = os.listdir(label_dir)
    print("Total {} labels".format(len(label_files)))

    label_paths = [os.path.join(label_dir, file) for file in label_files]
    sample_path = random.choice(label_paths)
    print("pick label sample {}".format(os.path.basename(sample_path)))

    sample_label = cv2.imread(sample_path)
    h, w, c = sample_label.shape
    # uint8类型
    print(sample_label.dtype)
    # 3通道
    print("height:{}; width:{}; channel:{}".format(h, w, c))

    # 发现有17种像素值，0-7，247-255
    print(np.unique(sample_label))
    show_image(sample_label, title='label', width=w, height=h)

    # TODO: 检查mask总共有几种像素值

    # TODO: 检查所有mask是否3通道

    # TODO: 检查所有标注图像大小与对应的原始图像是否一致


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
        mask = np.zeros((h, w, 3), dtype=np.uint8)
        print(mask.dtype)
        label = Image.fromarray(mask)
        label.save(label_path)


if __name__ == '__main__':
    # 将图像提取到对应目录
    # src = ['data01', 'data02', 'data03']
    # dst = 'Data/Train/Images'
    #
    # for s in src:
    #     extract_files(s, dst)

    # 图像文件所在目录
    root = 'Data/Train/Images'
    # 所有图像文件名称
    image_files = os.listdir(root)
    # 随机挑选一张样本做分析
    sample_file = random.choice(image_files)
    print("pick sample:{}".format(sample_file))
    sample_path = os.path.join(root, sample_file)
    sample = Image.open(sample_path)
    W, H = sample.size
    # RGB
    mode = sample.mode
    print("width:{}; height:{}; mode:{}".format(W, H, mode))
    # uint8类型
    sample_arr = np.asarray(sample)
    print(sample_arr.dtype)
    # sample.show()

    # image = cv2.cvtColor(np.asarray(sample), cv2.COLOR_RGB2BGR)
    # H, W, C = image.shape
    # show_image(image, title='sample', width=W, height=H)

    # 由于所有图像的宽、高不一致，因此统计下图像的尺寸
    # cal_sizes(root)

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

    # 统计样本的像素均值与方差
    # 所有图像文件的路径
    # image_paths = [os.path.join(root, image_file) for image_file in image_files]
    # pixel_calc(image_paths)

    # 为阴性样本生成标注mask
    # label_dir = 'Data/Train/Labels'
    # neg_paths = image_paths[:1000]
    # gen_mask_for_negatives(label_dir, neg_paths)

    # 标注图像分析
    # label_dir = 'Data/Train/Labels'
    # mask_analysis(label_dir)

    # 提交示例图像分析
    # result_dir = 'Data/Result/sample'
    # result_analysis(result_dir)
