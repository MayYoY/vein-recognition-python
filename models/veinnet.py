import torch.nn as nn
from skimage.feature import local_binary_pattern


class VeinNet(nn.Module):
    # 这个模型参数量很少，运算量很小，是为了方便没有GPU的同学做实验，
    # 效果未必最好，同学们可以根据自己的知识或通过学习《神经网络与深度学习》课程后优化模型
    def __init__(self, num_classes):
        super(VeinNet, self).__init__()
        # 以下定义四个卷积层，作用是通过训练后其卷积核具有提取静脉特征的能力
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        # self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0, groups=32)

        # 以下定义四个batch normalization层，作用是对中间数据做归一化处理
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        # self.bn5 = nn.BatchNorm2d(128)

        # 以下定义池化层，作用是对长和宽维度做下采样
        self.pool = nn.MaxPool2d(2, 2)

        # 以下定义激活层，作用是增加神经网络模型的非线性
        self.act = nn.LeakyReLU()

        # 以下定义最后的特征处理层，作用是将神经网络的三维矩阵特征变为一维向量特征后经过全连接层输出分类逻辑
        self.feature = nn.AdaptiveAvgPool2d(1)
        self.x2c = nn.Linear(64, num_classes)

    def forward(self, x):
        # 第一层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(x)
        x = self.act(x)
        # 第二层
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(x)
        x = self.act(x)
        # 第三层
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool(x)
        x = self.act(x)
        # 第四层
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool(x)
        x = self.act(x)
        # # 第五层
        # x = self.conv5(x)
        # x = self.bn5(x)
        # x = self.pool(x)
        # x = self.act(x)
        # 输出特征
        x = self.feature(x).view(-1, 64)
        c = self.x2c(x)
        return c

