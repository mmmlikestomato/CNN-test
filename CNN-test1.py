import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.optim import AdamW  # 直接导入 AdamW
# 或者
import torch.optim as optim     # 导入优化器模块，通过 optim.AdamW 调用
# 加载数据
csv_file_path ='augmented_data_5x.csv'
data_frame = pd.read_csv(csv_file_path)
print(1)
# 提取前256列数据并转换为张量
vectors = torch.tensor(data_frame.iloc[:, :256].values, dtype=torch.float32)

# 划分训练集和测试集索引
train_size = int(0.8 * len(vectors))
test_size = len(vectors) - train_size
train_indices, test_indices = random_split(range(len(vectors)), [train_size, test_size])

# 使用训练集的统计量进行归一化
train_vectors = vectors[train_indices]
train_mean = train_vectors.mean(dim=0, keepdim=True)
train_std = train_vectors.std(dim=0, keepdim=True)

vectors = (vectors - train_mean) / train_std

# 转置和重塑为16x16 若MLP 无需重构
vectors = vectors.view(-1, 16, 16).unsqueeze(1).permute(0, 1, 3, 2)  # 添加通道维度并进行转置


# 提取Mod_Type列并转换为数值标签
mod_types = data_frame['Mod_Type'].astype('category').cat.codes.values
labels = torch.tensor(mod_types, dtype=torch.long)

# 创建TensorDataset
dataset = TensorDataset(vectors, labels)
print(1)
# 创建训练集和测试集
train_dataset = TensorDataset(vectors[train_indices], labels[train_indices])
test_dataset = TensorDataset(vectors[test_indices], labels[test_indices])

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(1)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入和输出通道不匹配，使用1x1卷积调整维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out)
        return out

class OptimizedCNN(nn.Module):
    def __init__(self, num_classes=15):
        super(OptimizedCNN, self).__init__()
        self.in_channels = 16

        # 初始卷积层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # 残差块组
        self.layer1 = self._make_layer(16, 16, stride=1)  # 第1组
        self.layer2 = self._make_layer(16, 32, stride=2)  # 第2组（下采样）
        self.layer3 = self._make_layer(32, 64, stride=2)  # 第3组（下采样）

        # 全局平均池化 + 全连接层
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_channels, out_channels, stride):
        """生成残差块"""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

# 使用示例


def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total
print(1)

device = 'cpu'

# 初始化模型、损失函数、优化器和调度器
model = OptimizedCNN(num_classes=15).to(device)  # device = 'cuda' 或 'cpu'
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

# 数据加载器（假设已定义 train_dataset 和 test_dataset）
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print(1)
# 训练参数
num_epochs = 50
best_test_accuracy = 0.0
early_stop_patience = 5  # 早停阈值
early_stop_counter = 0
print(1)
# 记录训练过程
train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []

for epoch in range(num_epochs):
    # 训练阶段
    print(12)
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # 测试阶段
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    test_loss = running_loss / len(test_loader)
    test_accuracy = correct / total
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    # 学习率调度
    scheduler.step()

    # 打印信息
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f}")
    print(f"Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

    # 保存最佳模型（基于测试准确率）
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        torch.save(model.state_dict(), "best_model.pth")



print("Training complete. Best Test Accuracy:", best_test_accuracy)
torch.save(model.state_dict(), 'model_10dB-optimized.pth')
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.show()

plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.legend()
plt.show()
#2222222222import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 计算混淆矩阵并归一化
all_labels = []
all_predictions = []

model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted.numpy())

# 计算归一化后的混淆矩阵（按行归一化，每个类别总和为1）
cm = confusion_matrix(all_labels, all_predictions, normalize='true')

# 获取类别标签
class_labels = data_frame['Mod_Type'].astype('category').cat.categories

# 创建DataFrame保存归一化数据
cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)

# 保存归一化后的混淆矩阵数据
cm_df.to_csv("normalized_confusion_matrix.csv", float_format="%.4f")

# 绘制归一化后的混淆矩阵
plt.figure(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(
    include_values=True,
    cmap=plt.cm.Blues,
    xticks_rotation=45,
    values_format=".2f",  # 显示两位小数
)
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.savefig("normalized_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

