"""
    Pytorch的框架，具有非常好的弹性
    1. Prepare dataset
    2. Design model using Class
    3. Construct loss and optimizer
    4. Training cycle
"""

import torch
# 1. Prepare dataset
# 根据输入去定义w
# 张量需要输入二维矩阵，行代表item，列代表feature
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])
# x是3*1 y是3*1则
# w应该是1*3的矩阵 b应该是3*1的矩阵

# w = torch.Tensor([[],[],[]])
# b = torch.Tensor([[]])
# 最终loss一定要是一个标量，否则怎么求偏导


"""
以下就是pytorch的模板
必须有
__init__函数和forward函数

__init__中实现super继承
此外再构造你的网络节点

forward中实现返回预测值
"""
# 2. Design model using Class
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # 1输入1输出
        self.linear = torch.nn.Linear(1, 1)

    # forward会被__call__()调用
    # 调用LinerModel()即可求前馈
    def forward(self, x):
        # 输入线性运算单元，输出预测值
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()

# 3. Construct loss and optimizer
# 构造损失函数，使用均方误差，但是不求均值
criterion = torch.nn.MSELoss(size_average=False)
# 优化器，对模型中的所有神经元优化，其中学习率为0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

"""
    1. 求y_hat
    2. 求loss
    3. zero_grad
    4. backward
    5. step
"""

# 4. Training cycle
for epoch in range(1000):
    # forward
    y_pred = model(x_data)
    # loss是一个损失对象
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    # 所有权重归零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 优化器更新
    optimizer.step()


print("w =", model.linear.weight.item())
print("b =", model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print("y_pre =", y_test.item())








