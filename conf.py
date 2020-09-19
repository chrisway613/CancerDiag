# 是否使用GPU
GPU = True
# GPU id
DEVICE_ID = [0, 1]

# 数据集目录
DATA_DIR = 'Data'
# 可视化结果目录
VISUAL_DIR = 'Visual'
# 验证集划分比例
SPLIT_RATIO = .2
# 记录随机数种子
RANDOM_SEED = 'seed.txt'
# 划分后训练集和验证集图像与标签的路径
DS_PATHS = 'Data/paths/ds_path.db'

# 图像输入网络时缩放的尺寸 --（宽，高）
INPUT_SIZE = (744, 512)
# 中规模尺寸训练
MID_SIZE = (652, 448)
# 小尺寸训练
TINY_SIZE = (372, 256)

# 原始训练集（划分前）原图像像素均值与标准差(RGB)
TRAIN_MEAN = [208.644, 184.249, 206.240]
TRAIN_STD = [54.267, 77.503, 51.150]
# 训练集缩放后图像像素均值与标准差(RGB)
TRAIN_SCALE_MEAN = [.773, .619, .751]
TRAIN_SCALE_STD = [.220, .304, .202]
# 训练集缩放后阳性样本像素均值与标准差(RGB)
TRAIN_SCALE_POS_MEAN = [.767, .626, .751]
TRAIN_SCALE_POS_STD = [.233, .326, .215]
# 验证集缩放后图像像素均值与标准差(RGB)
EVAL_SCALE_MEAN = [.764, .623, .749]
EVAL_SCALE_STD = [.233, .326, .214]

# 测试集图像像素均值与标准差(RGB)
TEST_MEAN = [209.849, 188.015, 208.279]
TEST_STD = [53.722, 76.320, 50.763]

EPOCHS = 300
WARM_UP_EPOCH = 5
BATCH_SIZE = 8
BASE_LR = 5e-3
MIN_LR = 1e-4
NUM_CLASSES = 1
# 预测概率阀值
THRESH = .5
# 是否使用预训练权重
PRETRAINED = False
# 模型权重文件目录
CHECKPOINT = 'Weights'
# 是否使用同步BatchNorm
SYN_BN = False

# 打印log的迭代周期
LOG_CYCLE = 3
# 每10次迭代可视化预测概率图
VIS_CYCLE = 10
# 训练过程中进行评估验证的周期
TIME_TO_EVAL = 10

# 是否进行小样本训练
TINY_TRAIN = False
# 少样本模式下的样本总数量
TINY_NUM = 4
# 少样本训练模式下的批次大小
TINY_BATCH_SIZE = 2
