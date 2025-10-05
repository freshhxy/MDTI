import numpy as np  # 导入 NumPy 库，用于数值计算
import torch  # 导入 PyTorch 库，用于深度学习和张量操作
from torch import nn  # 从 PyTorch 中导入神经网络模块
from torch.nn import init  # 从 PyTorch 中导入初始化模块，用于权重初始化
from torch.nn import functional as F  # 导入 PyTorch 的函数式 API，用于在神经网络中应用各种功能

# 定义双重注意力模块 (DoubleAttention)
class DoubleAttention(nn.Module):

    def __init__(self, in_channels, c_m=128, c_n=128, reconstruct=True):
        super().__init__()

        # 初始化输入参数
        # in_channels: 输入的通道数
        # c_m: 第一个特征映射的通道数，默认为 128
        # c_n: 第二个特征映射的通道数，默认为 128
        # reconstruct: 是否重新构建输出的通道数，默认为 True
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.c_m = c_m
        self.c_n = c_n
        # 定义三个 1x1 的卷积层
        self.convA = nn.Conv2d(in_channels, c_m, 1)  # 用于计算特征 A
        self.convB = nn.Conv2d(in_channels, c_n, 1)  # 用于计算注意力映射 B
        self.convV = nn.Conv2d(in_channels, c_n, 1)  # 用于计算注意力向量 V

        # 如果需要重新构建输出通道数，定义一个 1x1 的卷积层
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size=1)
        # 初始化权重
        self.init_weights()

    # 定义权重初始化函数
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 对每个卷积层应用 He 正态分布初始化
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:  # 偏置初始化为 0
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # 对批量归一化层初始化
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # 对全连接层应用正态分布初始化
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    # 定义前向传播函数
    def forward(self, x):
        # 获取输入的形状
        b, c, h, w = x.shape
        assert c == self.in_channels  # 确保输入的通道数与定义的通道数一致
        # 通过三个卷积层计算特征 A、注意力映射 B 和注意力向量 V
        A = self.convA(x)  # 特征 A 的形状为 (b, c_m, h, w)
        B = self.convB(x)  # 注意力映射 B 的形状为 (b, c_n, h, w)
        V = self.convV(x)  # 注意力向量 V 的形状为 (b, c_n, h, w)
        # 重塑特征 A 为 (b, c_m, h*w)
        tmpA = A.view(b, self.c_m, -1)
        # 重塑并应用 softmax 到注意力映射 B，得到注意力权重，形状为 (b, c_n, h*w)
        attention_maps = F.softmax(B.view(b, self.c_n, -1), dim=-1)
        # 重塑并应用 softmax 到注意力向量 V，得到注意力权重，形状为 (b, c_n, h*w)
        attention_vectors = F.softmax(V.view(b, self.c_n, -1), dim=-1)
        # 第一步：特征门控
        # 计算特征 A 与注意力映射 B 的批量矩阵乘法，得到全局描述符，形状为 (b, c_m, c_n)
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))
        # 第二步：特征分布
        # 将全局描述符与注意力向量 V 相乘，得到新的特征映射 Z，形状为 (b, c_m, h*w)
        tmpZ = global_descriptors.matmul(attention_vectors)
        # 重塑 Z 为 (b, c_m, h, w)
        tmpZ = tmpZ.view(b, self.c_m, h, w)

        # 如果需要重新构建输出通道数，应用卷积层
        if self.reconstruct:
            tmpZ = self.conv_reconstruct(tmpZ)
        # 返回计算后的输出
        return tmpZ

# 测试代码块
if __name__ == '__main__':
    # 创建 DoubleAttention 模块实例，输入通道数为 32
    input = torch.randn(64, 32, 7, 7)  # 随机生成输入张量，形状为 (64, 32, 7, 7)
    a2 = DoubleAttention(32)
    output = a2(input)
    # 打印输出张量的形状，验证输出是否正确
    print(output.shape)
