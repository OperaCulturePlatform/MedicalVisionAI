import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import torch
import cv2
import numpy as np
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import LabelEncoder


def remove_black_borders(image, radius=None):
    """
    去除图像中的黑边，仅保留中间的圆形区域，并应用多种增强方法。

    Args:
        image (PIL.Image): 输入的PIL图像。
        radius (int): 圆形掩码的半径。如果为 None，自动取较短边的一半。

    Returns:
        PIL.Image: 去黑边并增强后的图像。
    """
    # 将 PIL 图像转换为 NumPy 数组，方便后续处理
    image = np.array(image)

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 使用阈值检测黑色区域，得到一个二值化的掩码
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 计算非零区域的最小外接圆的半径
    points = np.column_stack(np.where(mask > 0))  # 获取非零点的坐标
    center, radius = cv2.minEnclosingCircle(points)  # 获取最小外接圆的中心和半径

    # 将 radius 限制在图像的一半
    radius = min(radius, min(image.shape[:2]) // 2)
    # 创建圆形掩码
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), 255, thickness=-1)
    # 使用圆形掩码将黑边去除
    result = cv2.bitwise_and(image, image, mask=mask)

    # 1. CLAHE增强
    lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # 调整 clipLimit 和 tileGridSize
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    result = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # 2. Gamma校正
    result = gamma_correction(result, gamma=1.2)

    # 3. Histogram Equalization（直方图均衡化）
    result = histogram_equalization(result)
    # 返回最终增强后的图像
    return Image.fromarray(result)


def gamma_correction(image_array, gamma=1.2):
    """
    应用 Gamma 校正，调整图像亮度范围。

    Args:
        image (numpy.ndarray): 输入的图像。
        gamma (float): Gamma 校正系数，值大于1增强亮度，值小于1增加对比度。

    Returns:
        numpy.ndarray: 校正后的图像。
    """
    return np.array(255 * (image_array / 255) ** gamma, dtype=np.uint8)


def histogram_equalization(image):
    """
    对图像进行直方图均衡化。

    Args:
        image (numpy.ndarray): 输入的图像。

    Returns:
        numpy.ndarray: 均衡化后的图像。
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 应用直方图均衡化
    equalized = cv2.equalizeHist(gray)
    # 将均衡化后的图像合并回RGB
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)


class EyeDataset(Dataset):
    def __init__(self, dataset_dir, label_file, transform=None):
        """
        自定义数据集类，加载双目图像并合并为一个输入。

        Args:
            dataset_dir (str): 数据集目录。
            label_file (str): 标签文件路径。
            transform (torchvision.transforms.Compose): 图像预处理方法。
        """
        self.dataset_dir = dataset_dir

        # 读取标签文件
        self.labels = pd.read_excel(label_file)
        print("Columns in label file:", self.labels.columns)

        # 确保 'ID' 列存在
        if 'ID' in self.labels.columns:
            self.labels.rename(columns={'ID': 'id'}, inplace=True)

        # 直接加载所有行数据，不做 split 筛选
        print(f"Loaded {len(self.labels)} samples from label file.")

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 获取图像文件名和标签
        row = self.labels.iloc[idx]
        left_img_path = os.path.join(self.dataset_dir, row['Left-Fundus'])
        right_img_path = os.path.join(self.dataset_dir, row['Right-Fundus'])

        # 检查文件是否存在
        if not os.path.exists(left_img_path):
            raise FileNotFoundError(f"Left image file not found: {left_img_path}")
        if not os.path.exists(right_img_path):
            raise FileNotFoundError(f"Right image file not found: {right_img_path}")

        # 加载图像并去黑边
        # left_image = remove_black_borders(Image.open(left_img_path).convert("RGB"))
        # right_image = remove_black_borders(Image.open(right_img_path).convert("RGB"))

        left_image = Image.open(left_img_path).convert("RGB")
        right_image = Image.open(right_img_path).convert("RGB")

        # 调试输出图像类型
        # print(f"Left image type: {type(left_image)}")
        # print(f"Right image type: {type(right_image)}")

        # 图像预处理
        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        # 合并双目图像
        combined_image = torch.cat([left_image, right_image], dim=0)

        # 获取标签（最后的分类列）
        label = row[['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']].values.astype("float32")
        return combined_image, torch.tensor(label)


def train_check_augmentation(train_loader):
    images, labels = next(iter(train_loader))
    plt.figure(figsize=(12, 8))

    for i in range(6):
        plt.subplot(2, 3, i + 1)

        # 转换为 numpy 数组并调整维度为 (height, width, channels)
        img = images[i].numpy().transpose(1, 2, 0)

        # 反归一化操作：分别处理前 3 个通道和后 3 个通道
        img[:, :, :3] = img[:, :, :3] * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # 左眼
        img[:, :, 3:] = img[:, :, 3:] * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # 右眼

        # 拼接左右眼图像
        combined_eye = np.clip(img[:, :, :3], 0, 1)  # 左眼
        combined_eye_right = np.clip(img[:, :, 3:], 0, 1)  # 右眼
        combined_img = np.hstack((combined_eye, combined_eye_right))  # 水平拼接左右眼

        plt.imshow(combined_img)
        plt.title(f"Label: {labels[i].argmax()}")  # 假设标签是独热编码形式
    plt.savefig("result/train_augmentation_check.png")


def val_check_augmentation(train_loader):
    images, labels = next(iter(train_loader))
    plt.figure(figsize=(12, 8))

    for i in range(6):
        plt.subplot(2, 3, i + 1)

        # 转换为 numpy 数组并调整维度为 (height, width, channels)
        img = images[i].numpy().transpose(1, 2, 0)

        # 反归一化操作：分别处理前 3 个通道和后 3 个通道
        img[:, :, :3] = img[:, :, :3] * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # 左眼
        img[:, :, 3:] = img[:, :, 3:] * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # 右眼

        # 拼接左右眼图像
        combined_eye = np.clip(img[:, :, :3], 0, 1)  # 左眼
        combined_eye_right = np.clip(img[:, :, 3:], 0, 1)  # 右眼
        combined_img = np.hstack((combined_eye, combined_eye_right))  # 水平拼接左右眼

        plt.imshow(combined_img)
        plt.title(f"Label: {labels[i].argmax()}")  # 假设标签是独热编码形式
    plt.savefig("result/val_augmentation_check.png")

def test_check_augmentation(train_loader):
    images, labels = next(iter(train_loader))
    plt.figure(figsize=(12, 8))

    for i in range(6):
        plt.subplot(2, 3, i + 1)

        # 转换为 numpy 数组并调整维度为 (height, width, channels)
        img = images[i].numpy().transpose(1, 2, 0)

        # 反归一化操作：分别处理前 3 个通道和后 3 个通道
        img[:, :, :3] = img[:, :, :3] * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # 左眼
        img[:, :, 3:] = img[:, :, 3:] * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # 右眼

        # 拼接左右眼图像
        combined_eye = np.clip(img[:, :, :3], 0, 1)  # 左眼
        combined_eye_right = np.clip(img[:, :, 3:], 0, 1)  # 右眼
        combined_img = np.hstack((combined_eye, combined_eye_right))  # 水平拼接左右眼

        plt.imshow(combined_img)
        plt.title(f"Label: {labels[i].argmax()}")  # 假设标签是独热编码形式
    plt.savefig("result/test_augmentation_check.png")


def update_class_weights(model, train_loader, criterion, device):
    """
    动态调整类别权重：根据每个类别的损失值，更新采样权重。

    Args:
        model (nn.Module): 训练的模型。
        train_loader (DataLoader): 训练数据加载器。
        criterion (nn.Module): 损失函数。
        device (torch.device): 设备（CPU 或 GPU）。

    Returns:
        torch.Tensor: 更新后的类别权重。
    """
    model.eval()
    class_loss = torch.zeros(8).to(device)  # 假设有8个类别

    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # 计算每个类别的损失（按类别标签进行损失的加权）
            loss = criterion(outputs, labels)
            for i in range(len(labels)):
                class_loss[labels[i]] += loss[i]

    # 基于损失调整权重（权重与损失成反比）
    class_weights = 1.0 / (class_loss + 1e-4)  # 防止除零
    class_weights = class_weights / class_weights.sum()  # 归一化权重
    return class_weights


def get_dynamic_sampler(class_weights, train_loader):
    """
    根据动态权重获取加权采样器。

    Args:
        class_weights (torch.Tensor): 类别权重。
        train_loader (DataLoader): 训练数据加载器。

    Returns:
        WeightedRandomSampler: 加权采样器。
    """
    labels = train_loader.dataset.labels
    sample_weights = class_weights[labels.argmax(axis=1)]  # 根据类别权重获取样本的权重
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return sampler


class BalancedDataset(Dataset):
    """
    通过过采样和欠采样方法来平衡数据集。
    """
    def __init__(self, dataset, oversample_rate=1.0, undersample_rate=1.0):
        print(self.dataset.labels.head())  # 打印标签数据的前几行
        print(self.dataset.labels.shape)  # 打印标签数组的形状
        self.dataset = dataset
        self.oversample_rate = oversample_rate
        self.undersample_rate = undersample_rate

        # 确保标签是数值型（如果是字符串类型，则使用 LabelEncoder 转换）
        if isinstance(self.dataset.labels.iloc[0], str):
            label_encoder = LabelEncoder()
            self.dataset.labels = label_encoder.fit_transform(self.dataset.labels)

        # 如果标签是 one-hot 编码的形式，使用 argmax 提取出类别的索引
        if self.dataset.labels.ndim > 1:  # 检查标签是否是二维的（即 one-hot 编码）
            # 转换为 numpy 数组后使用 argmax 提取每行的类别索引
            labels = self.dataset.labels.to_numpy().argmax(axis=1)
        else:
            # 如果标签已经是整数类型（如 [0, 1, 2, ...]），直接使用它们
            labels = self.dataset.labels.to_numpy()

        # 获取每个类别的样本数量
        self.class_counts = Counter(labels)  # 计算每个类别的样本数量
        self.max_class_count = max(self.class_counts.values())

        # 根据过采样率或欠采样率调整样本数量
        self.sample_indices = []
        for idx, label in enumerate(labels):
            count = self.class_counts[label]
            if count < self.max_class_count * self.undersample_rate:
                self.sample_indices.append(idx)  # 欠采样：选择部分多数类样本
            elif count > self.max_class_count * self.oversample_rate:
                self.sample_indices.extend([idx] * (count // self.max_class_count))  # 过采样：重复少数类样本

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self.dataset[self.sample_indices[idx]]


def load_data(dataset_dir, label_file, model, criterion, device, batch_size=32, img_size=224):
    """
    加载训练、验证和测试集。

    Args:
        dataset_dir (str): 数据集目录。
        label_file (str): 标签文件路径。
        batch_size (int): 批量大小。
        img_size (int): 图像尺寸。

    Returns:
        DataLoader: 训练、验证和测试集的数据加载器。
    """

    base_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # 统一调整大小
        transforms.Grayscale(num_output_channels=3),  # 转为灰度图，方便后续处理
        transforms.Lambda(lambda img: Image.fromarray(gamma_correction(np.array(img), gamma=1.2))),  # Gamma校正增强亮度
        transforms.Lambda(lambda img: Image.fromarray(np.array(img) - np.min(np.array(img)))),  # 对比度增强
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    eval_transform = base_transform
    train_transform = base_transform

    # 加载训练集、验证集和测试集
    train_dataset = EyeDataset(
        os.path.join(dataset_dir, "train"),
        label_file,
        split='train',
        transform=train_transform
    )
    val_dataset = EyeDataset(
        os.path.join(dataset_dir, "val"),
        label_file,
        split='val',
        transform=eval_transform
    )
    test_dataset = EyeDataset(
        os.path.join(dataset_dir, "test"),
        label_file,
        split='test',
        transform=eval_transform
    )

    labels = train_dataset.labels[['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']].values
    class_counts = labels.sum(axis=0)
    class_weights = 1. / class_counts
    sample_weights = class_weights[labels.argmax(axis=1)]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size,sampler=sampler, num_workers=4,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    train_check_augmentation(train_loader)
    val_check_augmentation(val_loader)
    test_check_augmentation(test_loader)

    return train_loader, val_loader, test_loader