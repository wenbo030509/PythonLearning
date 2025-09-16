import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 1. 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 第一个全连接层：输入是28x28=784（类似MNIST手写数字的尺寸）
        self.fc1 = nn.Linear(784, 256)
        # 第二个全连接层
        self.fc2 = nn.Linear(256, 128)
        # 输出层：10个类别（比如0-9数字分类）
        self.fc3 = nn.Linear(128, 10)
        # Dropout层：防止过拟合
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # 将输入展平：(batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.view(-1, 784)
        # 第一个隐藏层，使用ReLU激活函数
        x = F.relu(self.fc1(x))
        # 应用dropout
        x = self.dropout(x)
        # 第二个隐藏层
        x = F.relu(self.fc2(x))
        # 输出层，不使用激活函数（因为后面会用CrossEntropyLoss）
        x = self.fc3(x)
        return x

# 2. 检查并设置设备（优先使用Mac的GPU）
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}")

# 3. 初始化模型并移动到指定设备
model = SimpleNN().to(device)

# 4. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 适用于分类问题
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# 5. 创建随机测试数据（模拟100张28x28的灰度图像）
# 输入: (batch_size, channels, height, width)
inputs = torch.randn(100, 1, 28, 28).to(device)
# 随机生成标签（0-9之间）
labels = torch.randint(0, 10, (100,)).to(device)

# 6. 训练一个批次的示例
model.train()  # 设置为训练模式
optimizer.zero_grad()  # 清零梯度

# 前向传播
outputs = model(inputs)
# 计算损失
loss = criterion(outputs, labels)
# 反向传播（计算梯度）
loss.backward()
# 更新权重
optimizer.step()

print(f"训练示例 - 损失值: {loss.item():.4f}")

# 7. 测试模型（推理模式）
model.eval()  # 设置为评估模式
with torch.no_grad():  # 禁用梯度计算，节省内存和计算资源
    test_inputs = torch.randn(10, 1, 28, 28).to(device)
    outputs = model(test_inputs)
    # 获取预测结果（每个样本概率最大的类别）
    _, predicted = torch.max(outputs, 1)
    print(f"预测结果: {predicted.tolist()}")
