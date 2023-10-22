import torch
import matplotlib.pyplot as plt


def forward(x):
    return w1 * (x ** 2) + w2 * x + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2




x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = torch.tensor([1.0])
w1.requires_grad = True
w2 = torch.tensor([1.0])
w2.requires_grad = True
b = torch.tensor([1.0])
b.requires_grad = True

mse_list = []
"""
    1.调用loss计算前馈
    2.调用backward计算反向
    3.修改权值w
"""
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        mse_list.append(l.item())
        l.backward()

        # 这是只改里面的值了，不需要重新构建关系图了
        w1.data = w1 - 0.01 * w1.grad.data
        w2.data = w2 - 0.01 * w2.grad.data
        b.data = b - 0.01 * b.grad.data

        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()
    print("progress:", epoch, l.item())


print("x = 4, y =", forward(4).item())
plt.plot(mse_list)
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.title("loss in each epoch")
plt.show()