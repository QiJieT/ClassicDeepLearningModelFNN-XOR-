import torch
import torch.nn as nn


# 定义网络结构
class XOR_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 2)  # 输入层到隐藏层
        self.output = nn.Linear(2, 1)  # 隐藏层到输出层
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))  # 隐藏层使用ReLU
        x = self.sigmoid(self.output(x))
        return x


# 定义数据和训练参数
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float)
model = XOR_Net()
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), lr=0.3)

# 训练循环
for epoch in range(1000):
    y_pred = model(X)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 测试结果
print(model(X).detach().numpy().round())
