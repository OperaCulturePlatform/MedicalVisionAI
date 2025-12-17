import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 384
NUM_CLASSES = 8

# 确保图像是 RGB 格式且大小正确
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # 确保图像为 RGB 格式
    image = image.resize((IMG_SIZE, IMG_SIZE))  # 调整图像大小
    image = np.array(image) / 255.0  # 归一化到[0, 1]范围
    image = torch.tensor(image).permute(2, 0, 1).float()  # 转换为Tensor (C, H, W)
    return image

class DualStreamDenseNetModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # 使用 DenseNet121 作为主干网络
        self.left_stream = self.get_densenet121(in_channels=3, num_classes=num_classes)
        self.right_stream = self.get_densenet121(in_channels=3, num_classes=num_classes)

        # 去掉 DenseNet 的全连接层
        self.left_stream.classifier = nn.Identity()
        self.right_stream.classifier = nn.Identity()

        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(2 * 1024, 2048),  # 输入维度 2 * 1024 = 2048
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        # 分类层
        self.classifier = nn.Linear(1024, num_classes)

    def get_densenet121(self, in_channels, num_classes):
        """
        获取预训练的 DenseNet121 模型，并修改输入层。
        """

        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        # 修改输入通道为3
        model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model

    def forward(self, x):
        # 通过两个分支处理左眼和右眼的输入
        left = self.left_stream(x[:, :3, :, :])  # 左眼图像 (batch_size, 3, height, width)
        right = self.right_stream(x[:, 3:, :, :])  # 右眼图像 (batch_size, 3, height, width)
        # 进行池化（四维张量）
        if left.dim() > 2:
            left = F.adaptive_avg_pool2d(left, (1, 1)).view(left.size(0), -1)  # (batch_size, 2048)
        if right.dim() > 2:
            right = F.adaptive_avg_pool2d(right, (1, 1)).view(right.size(0), -1)  # (batch_size, 2048)
        # 特征拼接
        fused = torch.cat([left, right], dim=1)  # (batch_size, 2048)
        # 特征融合
        fused = self.fusion(fused)
        # 分类
        output = self.classifier(fused)
        return output


def load_model():
    model = DualStreamDenseNetModel(num_classes=NUM_CLASSES).to(DEVICE)
    # 确保模型权重文件路径正确
    state_dict = torch.load('models/DenseNet121_best.pth', map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()  # 设置模型为评估模式
    return model


def predict(model, image_left, image_right):
    with torch.no_grad():
        # 组合左眼和右眼图像为(batch_size, 6, height, width)
        input_tensor = torch.cat([image_left.unsqueeze(0), image_right.unsqueeze(0)], dim=1).to(DEVICE)
        output = model(input_tensor)  # 前向传播
        _, predicted = torch.max(output, 1)  # 获取预测的类别
        return predicted.item()


def predict_main(test_images_left, test_images_right):
    model = load_model()

    predictions = []
    for left_img, right_img in zip(test_images_left, test_images_right):
        image_left = preprocess_image(left_img)
        image_right = preprocess_image(right_img)
        prediction = predict(model, image_left, image_right)
        predictions.append(prediction)

    return predictions[0]  # 返回第一张图的预测结果
