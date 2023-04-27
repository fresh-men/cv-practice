import math
from typing import Any

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F_torch
from torchvision import models
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

__all__ = [
    "SRResNet", "Discriminator",
    "srresnet_x4", "discriminator", "content_loss",
]

# SRResNet 模型定义
class SRResNet(nn.Module):
    def __init__(
            self,
            in_channels: int,  # 输入图像的通道数
            out_channels: int,  # 输出图像的通道数
            channels: int,  # 特征图的通道数
            num_rcb: int,  # RCB 残差块的数量
            upscale_factor: int  # 放大倍数
    ) -> None:
        super(SRResNet, self).__init__()
        # 低频信息提取层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, (9, 9), (1, 1), (4, 4)),
            nn.PReLU(),
        )

        # 高频信息提取块
        trunk = []
        for _ in range(num_rcb):
            trunk.append(_ResidualConvBlock(channels))
        self.trunk = nn.Sequential(*trunk)

        # 高频信息线性融合层
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

        # 放大块
        upsampling = []
        if upscale_factor == 2 or upscale_factor == 4 or upscale_factor == 8:
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling.append(_UpsampleBlock(channels, 2))
        elif upscale_factor == 3:
            upsampling.append(_UpsampleBlock(channels, 3))
        self.upsampling = nn.Sequential(*upsampling)

        # 重建块
        self.conv3 = nn.Conv2d(channels, out_channels, (9, 9), (1, 1), (4, 4))

        # 初始化神经网络权重
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # 支持 torch.script 函数
    def _forward_impl(self, x: Tensor) -> Tensor:
        out1 = self.conv1(x)
        out = self.trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)

        out = torch.clamp_(out, 0.0, 1.0)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)

# 判别器定义
class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            # 输入大小为 3 x 96 x 96
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # 输出大小为 64 x 48 x 48
            nn.Conv2d(64, 64, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # 输出大小为 128 x 24 x 24
            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # 输出大小为 256 x 12 x 12
            nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # 输出大小为 512 x 6 x 6
            nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # 输入的图像大小必须为 96
        assert x.shape[2] == 96 and x.shape[3] == 96, "Image shape must equal 96x96"

        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

# RCB 残差块定义
class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(_ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.rcb(x)
        out = torch.add(out, identity)

        return out

# 放大块
class _UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(_UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.upsample_block(x)

        return out

# 损失函数
class _ContentLoss(nn.Module):
    def __init__(
            self,
            feature_model_extractor_node: str,  # 特征提取网络节点名称
            feature_model_normalize_mean: list,  # 数据标准化均值
            feature_model_normalize_std: list  # 数据标准化标准差
    ) -> None:
        super(_ContentLoss, self).__init__()
        # 获取指定特征提取节点的名称
        self.feature_model_extractor_node = feature_model_extractor_node
        # 加载在 ImageNet 数据集上训练的 VGG19 模型。
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # 将 VGG19 模型的第三十六层输出作为内容损失。
        self.feature_extractor = create_feature_extractor(model, [feature_model_extractor_node])
        self.feature_extractor.eval()

        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Normalize(mean=feature_model_normalize_mean, std=feature_model_normalize_std),
        ])

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # 数据预处理
        x = self.transform(x)
        y = self.transform(y)
        # 提取特征
        x_features = self.feature_extractor(x)[self.feature_model_extractor_node]
        y_features = self.feature_extractor(y)[self.feature_model_extractor_node]

        loss = F_torch.mse_loss(x_features, y_features)

        return loss

def srresnet_x4(**kwargs: Any) -> SRResNet:
    model = SRResNet(upscale_factor=4, **kwargs)

    return model


def discriminator() -> Discriminator:
    model = Discriminator()

    return model


def content_loss(**kwargs: Any) -> _ContentLoss:
    content_loss = _ContentLoss(**kwargs)

    return content_loss


