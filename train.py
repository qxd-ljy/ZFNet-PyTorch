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
