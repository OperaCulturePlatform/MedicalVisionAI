import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from data_resize import crop_and_resize_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 384
NUM_CLASSES = 8


# -------------------------------
# 1. 定义用于双流模型的 DualGradCAM 类
# -------------------------------
class DualGradCAM:
    def __init__(self, model, branch, target_layer):
        """
        model: 双流模型（DualStreamDenseNetModel）
        branch: 指定需要解释的分支，'left' 或 'right'
        target_layer: 对应分支上用于提取特征的目标卷积层，例如 model.left_stream.features[-1]
        """
        self.model = model
        self.branch = branch
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # 注册钩子捕获前向输出和反向梯度
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        """
        input_tensor: 合并后的输入图像 Tensor，形状 (1, 6, H, W)
        class_idx: 要解释的类别索引
        """
        output = self.model(input_tensor)  # 前向传播（双流模型）
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)

        # 计算目标层梯度的全局平均池化得到通道权重
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # 加权求和得到热力图
        grad_cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        grad_cam = F.relu(grad_cam)
        grad_cam = F.interpolate(grad_cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        grad_cam = grad_cam.squeeze().cpu().numpy()
        return grad_cam


# -------------------------------
# 2. 修改后的 get_grad_cam_dual 函数
# -------------------------------
def get_grad_cam_dual(model, target_layer, left_image_path, right_image_path, branch='left'):
    """
    model: 已加载且处于评估模式的双流模型（DualStreamDenseNetModel）
    target_layer: 针对指定分支的目标层，例如 model.left_stream.features[-1] 或 model.right_stream.features[-1]
    left_image_path: 左眼图像路径
    right_image_path: 右眼图像路径
    branch: 'left' 或 'right'，决定生成哪边的 Grad-CAM
    """
    # 对左右眼图像分别进行裁剪和调整（你的crop_and_resize_image函数负责将图像裁剪为圆底384x384）
    crop_and_resize_image(left_image_path, left_image_path)
    crop_and_resize_image(right_image_path, right_image_path)

    # 定义预处理转换（注意：这里将图像调整为模型需要的尺寸，并归一化）
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # 加载左右眼图像
    left_img = Image.open(left_image_path).convert('RGB')
    right_img = Image.open(right_image_path).convert('RGB')
    left_tensor = transform(left_img)  # (3, H, W)
    right_tensor = transform(right_img)  # (3, H, W)

    # 合并左右眼图像得到 6 通道输入，shape: (1, 6, H, W)
    combined_tensor = torch.cat([left_tensor, right_tensor], dim=0).unsqueeze(0).to(DEVICE)

    # 先通过全模型获得预测类别
    output = model(combined_tensor)
    pred_class = output.argmax(dim=1).item()

    # 构建 DualGradCAM 对象（注意：target_layer应对应 branch）
    dual_grad_cam = DualGradCAM(model, branch, target_layer)
    cam_map = dual_grad_cam.generate(combined_tensor, pred_class)

    # 根据 branch 选择对应原图用于叠加
    if branch == 'left':
        original_image = cv2.imread(left_image_path)
    else:
        original_image = cv2.imread(right_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image, (IMG_SIZE, IMG_SIZE))

    # 归一化并转换为热力图
    heatmap = (cam_map - np.min(cam_map)) / (np.max(cam_map) - np.min(cam_map) + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)

    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.title(f"Dual Grad-CAM ({branch.capitalize()} branch, Predicted Class: {pred_class})")
    plt.axis('off')
    plt.savefig(f'./static/result/DualGradCAM_{branch}.png')
    plt.close()


# -------------------------------
# 3. 双流模型定义（与你的代码一致）
# -------------------------------
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
            nn.Linear(2 * 1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )
        # 分类层
        self.classifier = nn.Linear(1024, num_classes)

    def get_densenet121(self, in_channels, num_classes):
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model

    def forward(self, x):
        left = self.left_stream(x[:, :3, :, :])
        right = self.right_stream(x[:, 3:, :, :])
        if left.dim() > 2:
            left = F.adaptive_avg_pool2d(left, (1, 1)).view(left.size(0), -1)
        if right.dim() > 2:
            right = F.adaptive_avg_pool2d(right, (1, 1)).view(right.size(0), -1)
        fused = torch.cat([left, right], dim=1)
        fused = self.fusion(fused)
        output = self.classifier(fused)
        return output


# -------------------------------
# 4. 测试用例
# -------------------------------
def test_grad_cam_dual(left_image_path,right_image_path):
    # 加载模型并设置为评估模式
    model = DualStreamDenseNetModel(NUM_CLASSES).to(DEVICE)
    model.eval()
    # （这里假设模型权重已加载，如有需要请调用 load_state_dict 加载权重）

    # 选择目标层，例如：对左眼分支使用 DenseNet121 的最后一层卷积层
    left_target_layer = model.left_stream.features[-1]
    # 生成左眼 Grad-CAM 可视化
    get_grad_cam_dual(model, left_target_layer, left_image_path, right_image_path, branch='left')

    # 如果需要对右眼也进行解释，则选择右眼分支的目标层
    right_target_layer = model.right_stream.features[-1]
    get_grad_cam_dual(model, right_target_layer, left_image_path, right_image_path, branch='right')


if __name__ == '__main__':
    left_image_path = "./static/images/0_left.jpg"
    right_image_path = "./static/images/0_right.jpg"
    test_grad_cam_dual(left_image_path,right_image_path)
