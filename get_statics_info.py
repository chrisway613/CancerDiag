import cv2 as cv
import os
import numpy as np
# os.chdir('/home/stu/LJH/分割/')


def read_image_and_get_info(path):
    """
    function: 读取图片并统计白像素点和黑像素点的比值
    :parameter:path:图片路径
    :return:
        info:白像素/黑像素，白像素点的个数，黑像素点的个数
    """
    # 读取图片
    image = cv.imread(path)

    # 统计像素
    num_of_black = sum(sum(sum(image == 0)))
    num_of_white = sum(sum(sum(image == 255)))

    # 计算比例
    if num_of_black == 0:
        raise Exception('黑像素个数为0')

    # 计算白像素点和黑像素点的比值
    ration = num_of_white / num_of_black

    return [ration, num_of_white, num_of_black]


def get_images_info(path):
    """
    function:统计文件夹中图片的信息
    :param path: 文件夹的路径/文件的名称
    :return: None
    """
    # 用于存储比例信息
    white_ration_black = np.array([])
    # 用于存储像素点的个数
    num_of_white = np.array([])
    num_of_black = np.array([])

    # 获取所有图片路径
    images_name = path

    # 遍历所有的图片
    for image in images_name:
        # 计算白像素点和黑像素点的比值
        ra, wh, bl = read_image_and_get_info(image)
        white_ration_black = np.append(white_ration_black, ra)
        num_of_white = np.append(num_of_white, wh)
        num_of_black = np.append(num_of_black, bl)

    print('>Ration info:')
    # 计算比例的均值
    print('>>>mean:%.3f' % white_ration_black.mean())
    # 计算比例的方差
    print('>>>std:%.3f' % white_ration_black.std())

    print('>White pixel info:')
    # 计算白像素点的均值
    print('>>>mean:%.3f' % num_of_white.mean())
    # 计算白像素点的方差
    print('>>>std:%.3f' % num_of_white.std())

    print('>Black pixel info:')
    # 计算黑像素点的均值
    print('>>>mean:%.3f' % num_of_black.mean())
    # 计算黑像素点的方差
    print('>>>mean:%.3f' % num_of_black.std())


if __name__ == '__main__':
    root = 'Data/Train/Labels'
    image_files = os.listdir(root)

    # 获取阳性样本label路径
    pos_path = [os.path.join(root, filename)
                for filename in image_files if int(filename.split('_')[0].strip()) >= 2000]
    # 获取阴性样本label路径
    neg_path = [os.path.join(root, filename)
                for filename in image_files if int(filename.split('_')[0].strip()) < 2000]
    # 计算阳性label的信息
    print('---------------Info of positive:---------------')
    get_images_info(pos_path)

    # 计算阴性label的信息
    print('---------------Info of negative:---------------')
    get_images_info(neg_path)



