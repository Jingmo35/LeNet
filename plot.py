# 导入必要的库
from torchvision.datasets import FashionMNIST  # 从torchvision导入FashionMNIST数据集
from torchvision import transforms  # 导入图像预处理和变换模块
import torch.utils.data as Data  # 导入PyTorch数据加载工具
import numpy as np  # 导入数值计算库
import matplotlib.pyplot as plt  # 导入绘图库

# 加载FashionMNIST训练数据集
# FashionMNIST是一个包含10类时尚物品的图像数据集，常用于替代MNIST进行图像分类
train_data = FashionMNIST(
    root='./data',  # 数据集下载和存储的根目录
    train=True,  # 设置为True表示加载训练集，False表示加载测试集
    transform=transforms.Compose([  # 定义图像预处理流水线
        transforms.Resize(size=224),  # 将图像尺寸调整为224×224（通常用于适配预训练模型）
        transforms.ToTensor()  # 将PIL图像或numpy数组转换为PyTorch张量，并自动归一化到[0,1]范围
    ]),
    download=True  # 如果数据集不存在，则自动下载
)

# 创建数据加载器，用于批量加载数据
train_loader = Data.DataLoader(
    dataset=train_data,  # 指定要加载的数据集
    batch_size=64,  # 每个批次包含64个样本
    shuffle=True,  # 每个epoch开始时打乱数据顺序，防止模型学习到数据顺序
    num_workers=0  # 使用0个进程加载数据（0表示在主进程中加载）
)

# 从数据加载器中获取一个批次的数据进行可视化
for step, (b_x, b_y) in enumerate(train_loader):
    # step: 当前批次的索引
    # b_x: 包含64个图像数据的张量，形状为[64, 1, 224, 224]
    # b_y: 包含64个标签数据的张量，形状为[64]
    if step > 0:  # 只获取第一个批次的数据后就跳出循环
        break

# 将PyTorch张量转换为Numpy数组以便处理和可视化
batch_x = b_x.squeeze().numpy()  
# b_x.squeeze(): 移除张量中维度为1的维度，从[64,1,224,224]变为[64,224,224]
# .numpy(): 将PyTorch张量转换为Numpy数组

batch_y = b_y.numpy()  # 将标签张量转换为Numpy数组

class_label = train_data.classes  # 获取数据集的类别标签名称
# print(class_label)  # 如果取消注释，会打印所有类别名称

# 打印批次数据的形状信息
print("The size of batch in train data:", batch_x.shape)  
# 输出：The size of batch in train data: (64, 224, 224)
# 表示有64个样本，每个样本是224×224像素的图像

# 可视化当前批次的所有图像
plt.figure(figsize=(12, 5))  # 创建图形窗口，设置大小为12×5英寸

# 循环遍历批次中的每个样本（共64个）
for ii in np.arange(len(batch_y)):
    # 创建4行16列的子图布局，当前绘制第ii+1个子图
    plt.subplot(4, 16, ii + 1)
    
    # 显示图像
    plt.imshow(batch_x[ii, :, :],  # 选择第ii个样本的所有像素
               cmap=plt.cm.gray)  # 使用灰度色彩映射（因为FashionMNIST是灰度图像）
    
    # 设置子图标题为对应的类别名称
    plt.title(class_label[batch_y[ii]],  # 根据标签索引获取类别名称
              size=10)  # 设置标题字体大小为10
    
    plt.axis("off")  # 关闭坐标轴显示，使图像更清晰
    
    # 调整子图之间的水平间距为0.05
    plt.subplots_adjust(wspace=0.05)

# 显示图形
plt.show()