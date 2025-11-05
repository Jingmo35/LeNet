# 导入必要的库
import copy  # 导入深拷贝模块，用于复制模型参数
import time  # 导入时间模块，用于计算训练耗时
import torch  # 导入PyTorch深度学习框架
from torchvision.datasets import FashionMNIST  # 导入FashionMNIST数据集
from torchvision import transforms  # 导入图像预处理模块
import torch.utils.data as Data  # 导入数据加载工具
import numpy as np  # 导入数值计算库
import matplotlib.pyplot as plt  # 导入绘图库
from model import LeNet  # 从model.py文件中导入自定义的LeNet模型
import torch.nn as nn  # 导入PyTorch神经网络模块
import pandas as pd  # 导入数据分析库

# 定义数据预处理和划分函数
def train_val_data_process():
    """
    加载FashionMNIST数据集并进行训练集和验证集的划分
    返回训练数据加载器和验证数据加载器
    """
    # 加载FashionMNIST训练数据集
    train_data = FashionMNIST(root='./data',  # 数据集存储路径
                              train=True,  # 加载训练集
                              transform=transforms.Compose([  # 定义数据预处理流水线
                                  transforms.Resize(size=28),  # 调整图像尺寸为28
                                  transforms.ToTensor()  # 转换为张量并归一化到[0,1]
                              ]),
                              download=True)  # 如果数据集不存在则下载


    '''
    batch_size对应一种输入机制，它将图像先成批打包，再输入网络，batch_size这里对应的就是每次打包的照片有多少张，该参数设置得越高，训练速度会相应变快，但对应的gpu的压力会变大。
    num_worker对应的是几个线程同时进行工作，gpu比较弱设置小一点batch_size==32,num_worker==0(只有一个进程即主进程的意思)
    '''
    # 将训练数据集划分为训练集和验证集（80%训练，20%验证）
    train_data, val_data = Data.random_split(train_data, 
                                           [round(0.8*len(train_data)),  # 训练集大小
                                            round(0.2*len(train_data))])  # 验证集大小

    # 创建训练数据加载器
    train_dataloader = Data.DataLoader(dataset=train_data,  # 训练数据集
                                       batch_size=32,  # 每个批次32个样本
                                       shuffle=True,  # 打乱数据顺序
                                       num_workers=0)  # 使用0个进程加载数据，只有一个进程即主进程

    # 创建验证数据加载器
    val_dataloader = Data.DataLoader(dataset=val_data,  # 验证数据集
                                     batch_size=32,  # 每个批次32个样本
                                     shuffle=True,  # 打乱数据顺序
                                     num_workers=0)  # 使用0个进程加载数据

    return train_dataloader, val_dataloader  # 返回两个数据加载器



# 定义模型训练和验证过程
def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    """
    训练和验证模型
    参数:
        model: 要训练的模型
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器  
        num_epochs: 训练轮数
    返回:
        train_process: 包含训练过程的DataFrame
    """
    # 设定训练所用到的设备，有GPU用GPU没有GPU用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 使用Adam优化器，学习率为0.001。可以理解为改进的梯度下降法
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 损失函数为交叉熵函数，适用于多分类问题
    criterion = nn.CrossEntropyLoss()
    
    # 将模型放入到训练设备中
    model = model.to(device)
    
    # 复制当前模型的参数，用于保存最佳模型
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化训练过程记录变量
    best_acc = 0.0  # 最高准确度
    train_loss_all = []  # 训练集损失列表
    val_loss_all = []  # 验证集损失列表
    train_acc_all = []  # 训练集准确度列表
    val_acc_all = []  # 验证集准确度列表
    since = time.time()  # 记录训练开始时间


    # 开始训练循环
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))  # 打印当前epoch；字符串中的{}是占位符，会被format方法中的参数替换。例如，如果epoch是0，num_epochs是10，那么会打印：Epoch 0/9
        print("-"*10)  # 打印分隔线

        # 初始化每个epoch的统计变量
        train_loss = 0.0  # 训练集损失函数累计值
        train_corrects = 0  # 训练集正确预测数量

        val_loss = 0.0  # 验证集损失函数累计值
        val_corrects = 0  # 验证集正确预测数量

        train_num = 0  # 训练集样本数量
        val_num = 0  # 验证集样本数量

        # 训练阶段：对每一个mini-batch（批次）进行训练
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将数据移动到指定设备
            b_x = b_x.to(device)  # 特征数据放入训练设备
            b_y = b_y.to(device)  # 标签数据放入训练设备
            
            # 设置模型为训练模式（启用dropout和batch normalization）
            model.train()

            # 前向传播：输入一个batch，输出对应的预测
            output = model(b_x)
            
            # 查找每一行中最大值（概率最大）对应的索引，即预测的类别
            pre_lab = torch.argmax(output, dim=1)
            
            # 计算当前batch的损失函数
            loss = criterion(output, b_y)

            # 反向传播过程
            optimizer.zero_grad()  # 每一轮都需要将梯度初始化为0，防止梯度累计
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 根据反向传播梯度更新网络参数，起到降低loss函数计算值的作用

            # 累计统计信息
            train_loss += loss.item() * b_x.size(0)  # 累计损失（乘以batch大小）
            train_corrects += torch.sum(pre_lab == b_y.data)  # 累计正确预测数
            train_num += b_x.size(0)  # 累计训练样本数


            '''验证阶段'''

        # 验证阶段：在验证集上评估模型
        for step, (b_x, b_y) in enumerate(val_dataloader):
            # 将数据移动到验证设备
            b_x = b_x.to(device)  # 特征数据
            b_y = b_y.to(device)  # 标签数据
            # 设置模型为验证评估模式（禁用dropout和batch normalization）
            model.eval()
            # 前向传播（不计算梯度），输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每个batch的损失函数
            loss = criterion(output, b_y)

            # 累计统计信息
            val_loss += loss.item() * b_x.size(0)  # 累计验证损失，loss.item()是样本平均loss
            val_corrects += torch.sum(pre_lab == b_y.data)  # 累计正确预测数
            val_num += b_x.size(0)  # 累计验证样本数



        # 计算并保存验证集每次迭代的平均损失和准确率
        train_loss_all.append(train_loss / train_num)  # 训练集平均损失
        train_acc_all.append(train_corrects.double().item() / train_num)  # 训练集准确率
        
        val_loss_all.append(val_loss / val_num)  # 验证集平均损失
        val_acc_all.append(val_corrects.double().item() / val_num)  # 验证集准确率

        # 打印当前epoch的结果
        print("{} train loss:{:.4f} train acc: {:.4f}".format(
            epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss :{:.4f} val acc: {:.4f}".format(
            epoch, val_loss_all[-1], val_acc_all[-1]))



        # 如果当前验证准确率比历史最佳高，则保存模型
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]  # 更新最佳准确率
            best_model_wts = copy.deepcopy(model.state_dict())  # 保存模型参数

        # 计算并打印训练耗时
        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use//60, time_use%60))

  
    # 训练结束后，保存最佳模型参数到文件
    torch.save(best_model_wts, "./best_model.pth")

    # 创建DataFrame保存训练过程数据
    train_process = pd.DataFrame(data={
        "epoch": range(num_epochs),  # 训练轮数
        "train_loss_all": train_loss_all,  # 训练损失
        "val_loss_all": val_loss_all,  # 验证损失
        "train_acc_all": train_acc_all,  # 训练准确率
        "val_acc_all": val_acc_all,  # 验证准确率
    })

    return train_process  # 返回训练过程数据


# 定义可视化函数
def matplot_acc_loss(train_process):
    """
    可视化训练过程中的损失和准确率变化
    参数:
        train_process: 包含训练过程的DataFrame
    """
    # 创建图形窗口，设置大小
    plt.figure(figsize=(12, 4))
    
    # 第一个子图：损失函数变化
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()  # 显示图例
    plt.xlabel("epoch")  # x轴标签
    plt.ylabel("Loss")  # y轴标签
    
    # 第二个子图：准确率变化
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")  # x轴标签
    plt.ylabel("acc")  # y轴标签
    plt.legend()  # 显示图例
    plt.show()  # 显示图形


# 主程序入口
if __name__ == '__main__':
    # 实例化LeNet模型
    LeNet = LeNet()
    # 加载并处理数据集
    train_data, val_data = train_val_data_process()
    # 训练模型，获取训练过程数据
    train_process = train_model_process(LeNet, train_data, val_data, num_epochs=10)
    # 可视化训练过程
    matplot_acc_loss(train_process)