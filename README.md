        在本篇博客中，我们将通过两个主要部分来演示如何使用 PyTorch 实现 ZFNet，并在 MNIST 数据集上进行训练和测试。ZFNet（ZFNet）是基于卷积神经网络（CNN）的图像分类模型，广泛用于图像识别任务。

环境准备
        在开始之前，请确保你的环境已经安装了以下依赖：

pip install torch torchvision matplotlib tqdm
点击并拖拽以移动
一、训练部分：训练 ZFNet 模型
首先，我们需要准备训练数据、定义 ZFNet 模型，并进行模型训练。

1. 数据加载与预处理
MNIST 数据集由 28x28 的手写数字图像组成。我们将通过 torchvision.datasets 来加载数据，并进行必要的预处理。

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from zfnet import ZFNet  # 假设 ZFNet 定义在 zfnet.py 文件中
from tqdm import tqdm  # 导入 tqdm
from torch.cuda.amp import autocast, GradScaler  # 导入混合精度训练


def prepare_data(batch_size=128, num_workers=2, data_dir='D:/workspace/data'):
    """
    准备 MNIST 数据集并返回数据加载器
    :param batch_size: 批处理大小
    :param num_workers: 数据加载的工作线程数
    :param data_dir: 数据存储的目录
    :return: 训练数据加载器
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 正则化
    ])

    trainset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    return trainloader
点击并拖拽以移动

2. 初始化模型与优化器
在这里，我们将初始化模型和优化器。我们选择 Adam 优化器，并且为提高计算效率，我们采用混合精度训练。

def initialize_device():
    """
    初始化计算设备（GPU 或 CPU）
    :return: 计算设备
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def initialize_model(device):
    """
    初始化模型并移动到指定设备
    :param device: 计算设备
    :return: 初始化好的模型
    """
    model = ZFNet().to(device)  # 假设 ZFNet 是自定义模型
    return model


def initialize_optimizer(model, lr=0.001):
    """
    初始化优化器
    :param model: 需要优化的模型
    :param lr: 学习率
    :return: 优化器
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return optimizer
点击并拖拽以移动

3. 训练模型
使用训练数据进行训练，并且每训练一个 epoch 就更新一次进度条，同时使用混合精度训练来提高效率。

def train_model(model, trainloader, criterion, optimizer, num_epochs=5, device='cuda'):
    """
    训练模型
    :param model: 训练的模型
    :param trainloader: 数据加载器
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param num_epochs: 训练的轮数
    :param device: 计算设备
    """
    scaler = GradScaler()  # 用于自动缩放梯度

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # 使用 tqdm 包裹 DataLoader 来显示进度条
        with tqdm(trainloader, unit="batch", desc=f"Epoch {epoch + 1}/{num_epochs}") as tepoch:
            for inputs, labels in tepoch:
                # 直接将数据和标签移动到 GPU
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                # 混合精度前向和反向传播
                with autocast():  # 自动混合精度
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                # 反向传播与优化
                scaler.scale(loss).backward()  # 使用混合精度反向传播
                scaler.step(optimizer)  # 更新参数
                scaler.update()  # 更新缩放因子

                running_loss += loss.item()

                # 更新进度条显示
                tepoch.set_postfix(loss=running_loss / (tepoch.n + 1))

        # 打印每个 epoch 的平均损失
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")

    # 保存模型
    torch.save(model.state_dict(), 'zfnet_model.pth')
    print("Model saved as zfnet_model.pth")
点击并拖拽以移动

4. 主函数
在主函数中，我们会初始化设备、模型、损失函数，并启动训练过程。

if __name__ == '__main__':
    """
       主函数：组织所有步骤的执行
       """
    # 数据加载
    trainloader = prepare_data()

    # 设备选择
    device = initialize_device()

    # 模型初始化
    model = initialize_model(device)

    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()

    # 优化器初始化
    optimizer = initialize_optimizer(model)

    # 启动训练
    train_model(model, trainloader, criterion, optimizer, num_epochs=5, device=device)
点击并拖拽以移动
二、测试部分：评估 ZFNet 模型
训练完成后，我们将加载训练好的模型，并在测试集上评估其性能。

1. 加载和预处理数据
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from zfnet import ZFNet  # 假设 ZFNet 定义在 zfnet.py 文件中


def load_and_preprocess_data(batch_size=1000):
    """
    加载并预处理 MNIST 数据集
    :param batch_size: 数据加载的批次大小
    :return: 测试数据加载器
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 下载 MNIST 测试集
    testset = datasets.MNIST(root='D:/workspace/data', train=False, download=True, transform=transform)

    # 数据加载器
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return testloader
点击并拖拽以移动

2. 加载训练好的模型
def load_and_preprocess_data(batch_size=1000):
    """
    加载并预处理 MNIST 数据集
    :param batch_size: 数据加载的批次大小
    :return: 测试数据加载器
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 下载 MNIST 测试集
    testset = datasets.MNIST(root='D:/workspace/data', train=False, download=True, transform=transform)

    # 数据加载器
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return testloader


def load_trained_model(model_path='zfnet_model.pth'):
    """
    加载训练好的模型
    :param model_path: 模型文件路径
    :return: 加载的模型
    """
    model = ZFNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式
    return model
点击并拖拽以移动

3. 评估模型
def evaluate_model(model, testloader):
    """
    评估模型在测试集上的表现
    :param model: 训练好的模型
    :param testloader: 测试数据加载器
    :return: 模型准确率
    """
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy
点击并拖拽以移动

4. 可视化预测结果
def visualize_predictions(model, testloader, num_images=6):
    """
    可视化模型对多张测试图片的预测结果
    :param model: 训练好的模型
    :param testloader: 测试数据加载器
    :param num_images: 显示图像的数量
    """
    model.eval()
    data_iter = iter(testloader)
    images, labels = next(data_iter)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # 绘制结果
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes = axes.ravel()

    for i in range(num_images):
        ax = axes[i]
        img = images[i].numpy().transpose(1, 2, 0)  # 将 Tensor 转换为 NumPy 数组并转置为 HWC 格式
        ax.imshow(img.squeeze(), cmap='gray')  # squeeze 去除单通道维度
        ax.set_title(f"Pred: {predicted[i].item()} | Actual: {labels[i].item()}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
点击并拖拽以移动

5. 主函数
在测试阶段，我们加载模型并在测试数据集上评估它。

def main():
    """
    主函数，组织数据加载、模型加载、评估和可视化步骤
    """
    # 加载并预处理数据
    testloader = load_and_preprocess_data()

    # 加载训练好的模型
    model = load_trained_model()

    # 评估模型
    accuracy = evaluate_model(model, testloader)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # 可视化预测结果
    visualize_predictions(model, testloader, num_images=6)


if __name__ == '__main__':
    main()
点击并拖拽以移动

结语
通过本文的介绍，我们实现了一个基于 ZFNet 模型的图像分类任务，使用 PyTorch 对 MNIST 数据集进行训练与测试，并展示了如何进行混合精度训练以提高效率。在未来，你可以根据不同的任务修改模型结构、优化器或者训练策略，进一步提升性能。


完整项目
ZFNet-PyTorch: 使用 PyTorch 实现 ZFNet 进行 MNIST 图像分类
https://gitee.com/qxdlll/zfnet-py-torch
GitHub - qxd-ljy/ZFNet-PyTorch: 使用 PyTorch 实现 ZFNet 进行 MNIST 图像分类
使用 PyTorch 实现 ZFNet 进行 MNIST 图像分类. Contribute to qxd-ljy/ZFNet-PyTorch development by creating an account on GitHub.
https://github.com/qxd-ljy/ZFNet-PyTorch



