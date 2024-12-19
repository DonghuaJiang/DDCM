import torch
import numpy as np


def weights_init_normal(m):                                                                # 初始化参数
    classname = m.__class__.__name__                                                       # __class__.__name__：获得类名
    if classname.find("Conv") != -1:                                                       # 如果是卷积层
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)                                    # 对卷积权重进行初始化
        # hasattr：has attribute -> 具有属性
        if hasattr(m, "bias") and m.bias is not None:                                      # 存在偏置且偏置不为空
            torch.nn.init.constant_(m.bias.data, 0.0)                                      # 用0进行初始化
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)                                    # 利用正态分布进行赋值（均值，方差）
        torch.nn.init.constant_(m.bias.data, 0.0)                                          # 初始化偏置


class ResidualBlock(torch.nn.Module):                                                      # 残差块
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()                                              # 调用父类的方法进行初始化

        self.block = torch.nn.Sequential(                                                  # Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d
            torch.nn.ReflectionPad2d(1),                                                   # ReflectionPad2d()：一种边界填充方法，与常规的零填充相比， 填充内容来自输入
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=3),                      # 矩阵维度：130*130 -> 128*128
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.ReflectionPad2d(1),                                                   # 矩阵维度：128*128 -> 130*130
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=3),                      # 矩阵维度：130*130 -> 128*128
            torch.nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(torch.nn.Module):                                                    # 生成器网络结构
    def __init__(self, input_channels, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        # 初始化卷积块
        channels = input_channels
        out_channels = 64
        model_1 = [                                                                        # 通道数：3 -> 64
            torch.nn.ReflectionPad2d(2),                                                   # 矩阵维度：128*128 -> 132*132
            torch.nn.Conv2d(channels, out_channels, kernel_size=5),                        # 矩阵维度：132*132 -> 128*128
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        ]
        in_channels = out_channels

        # 下采样
        model_2 = []
        for _ in range(2):
            out_channels *= 2                                                                     # 通道数：64 -> 128 -> 256
            model_2 += [
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),   # 矩阵维度：128*128 -> 64*64 -> 32*32
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True)
            ]
            in_channels = out_channels

        # 残差块
        model_3 = []
        for _ in range(num_residual_blocks):                                               # 通道数：256 -> 256
            model_3 += [ResidualBlock(out_channels)]                                       # 矩阵维度：32*32 -> 32*32

        # 上采样
        model_4 = []
        in_channels = 2 * in_channels
        for _ in range(2):
            out_channels //= 2                                                             # 通道数：256 -> 128 -> 64
            model_4 += [
                torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                torch.nn.BatchNorm2d(out_channels),                                        # 矩阵维度：32*32 -> 64*64 -> 128*128
                torch.nn.ReLU(inplace=True)
            ]
            in_channels = out_channels

        # 输出层
        model_5 = [
            torch.nn.ReflectionPad2d(2),                                                   # 通道数：128*num_residual_blocks -> 3
            torch.nn.Conv2d(2 * in_channels, 3, kernel_size=5),                            # 矩阵的维度不改变
            torch.nn.Tanh()
        ]

        self.model_input = torch.nn.Sequential(*model_1)
        self.model_down = torch.nn.Sequential(*model_2)
        self.model_resn = torch.nn.Sequential(*model_3)
        self.model_up = torch.nn.Sequential(*model_4)
        self.model_output = torch.nn.Sequential(*model_5)

    def forward(self, x):
        x1 = self.model_input(x)
        x2 = self.model_down(x1)
        x3 = self.model_resn(x2)
        x4 = self.model_up(torch.cat([x2, x3], dim=1))
        x5 = self.model_output(torch.cat([x1, x4], dim=1))
        return x5