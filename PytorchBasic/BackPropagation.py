"""
    反向传播
    BackPropagation

    forward 正向计算函数
    backward 反向计算偏导 Chain Role
"""

"""
In Pytorch , Tensor is the important component
Tensor w has data and Grad two component
    Tensor:
        1. Data w
        2. Grad dl / dw
"""

import torch
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 将权重w定义为Tensor
w = torch.Tensor([1.0])
# 他是需要梯度的，必须加上需要计算梯度，默认是False
w.requires_grad = True

# 前馈
def forward(x):
    return x * w    # 计算结果会自动转化为Tensor，且是需要梯度的

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2    # 算损失值

"""
    一个Tensor有如下属性
    tensor.data是这个张量的值
    tensor.grad是这个张量的梯度
    tensor.item()就是这个标量
    .data返回的还是一个张量，只不过这个张量不再支持求梯度，尽作为数值去乘
    .item()返回的是一个标量
"""

"""
    1.调用loss计算前馈
    2.调用backward计算反向
    3.修改权值w
"""

mse_list = []
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        # 前馈算出损失值，前馈最后算出的结果是张量
        # Forward, Compute the loss
        l = loss(x, y)
        mse_list.append(l.item())
        # 张量自带backword，调用算出偏导
        l.backward()
        print('\tgrad:', x, y, w.grad.item())
        # 修改权值，更新时一定使用.data .item()
        w.data = w.data - 0.01 * w.grad.data
        # data.zero_ 把梯度数据清零，为什么要清零，把这一次求导清零，否则会把梯度累加
        w.grad.data.zero_()
    print("progress:", epoch, l.item())

plt.plot(mse_list)
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.title("loss in each epoch")
plt.show()