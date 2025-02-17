import torch
import torch.nn as nn


# 定义网络结构
# nn.Module 是 PyTorch 中所有神经网络模块的父类。
# 为构建神经网络提供了诸多便利的功能与属性。
class XOR_Net(nn.Module):
    def __init__(self):
        # 调用父类 nn.Module 的构造函数。
        # 这一步至关重要，因为 nn.Module 的构造函数会执行一系列必要的初始化操作，
        # 像模块的注册、参数管理等。
        # 要是不调用父类的构造函数，这些初始化操作就不会被执行，可能致使该神经网络模块无法正常工作。
        super().__init__()
        self.hidden = nn.Linear(2, 2)  # 输入层到隐藏层
        # nn.Linear 用于构建全连接层的类。
        # 第一个参数 2 表示输入特征的数量，意味着输入层有 2 个神经元。
        # 第二个参数 2 表示输出特征的数量，即隐藏层有 2 个神经元。
        self.output = nn.Linear(2, 1)  # 隐藏层到输出层
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))  # 隐藏层使用ReLU
        x = self.sigmoid(self.output(x)) # 输出层使用sigmoid函数
        return x


if __name__ == "__main__":
    # 手动设置权重（基于上述推导）
    model = XOR_Net()
    # tensor：意思为张量
    # 标量（零阶张量）：可以理解为一个单独的数，没有方向
    # 向量（一阶张量）：是一组按顺序排列的数，有方向和大小。
    # 矩阵（二阶张量）：是由行和列组成的二维数组。
    # 自动求导：现代深度学习框架如 PyTorch 和 TensorFlow 都支持对 Tensor 进行自动求导。
    # 这意味着在定义好神经网络的计算图后，框架可以自动计算损失函数关于模型参数的梯度，从而方便进行模型的训练
    model.hidden.weight.data = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    model.hidden.bias.data = torch.tensor([-0.5, 1.5])
    model.output.weight.data = torch.tensor([[-1.0, -1.0]])
    model.output.bias.data = torch.tensor([1.25])
    # 测试XOR输入
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
    predictions = model(X)
    print(predictions.detach().numpy().round())
    # detach() 是 torch.Tensor 对象的一个方法，
    # 它的作用是创建一个新的张量，这个新张量和原张量共享数据内存，但不会保留计算图中的梯度信息。
    # 在深度学习中，当我们只关心张量的值，而不需要进行反向传播计算梯度时，
    # 使用 detach() 可以避免不必要的梯度计算，同时防止在后续操作中意外修改原张量的梯度信息。
    # 我们只需要模型的预测结果，不需要对这些结果进行反向传播更新模型参数，这时就可以使用 detach() 方法。
    # numpy() 同样是 torch.Tensor 对象的方法，它的功能是将 PyTorch 的张量转换为 NumPy 数组
    # round() 是 NumPy 数组的方法，它会对数组中的每个元素进行四舍五入操作。在分类问题中，如果模型输出的是概率值，
    # 通过四舍五入可以将其转换为具体的类别标签，大于0.5为1，小于0.5为0


