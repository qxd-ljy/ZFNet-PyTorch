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
