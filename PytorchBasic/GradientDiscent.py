"""

    Optimization Problem
    We Use Gradient Descent

    1. Initial Guess, Random w
    2. Gradient Descent Update w\
    3. Repeat Gradient Descent

"""
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
# Initial Guess
w = 1.0

def forward(x):
    return x * w

# 求MSE
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

# 求梯度，这里我们求出了导数grad = 2 * x * (x * w - y)
def gradient(x, y):
    return 2 * x * (x * w - y)

mse_list = []
lr = 0.01
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        # 该点处的梯度
        grad = gradient(x, y)
        # 修改w
        w = w - lr * grad
        print("\tgrad: ", x, y, grad)
        # 求该点处的损失值
        loss_val = loss(x, y)
        mse_list.append(loss_val)
    # 输出本轮情况
    print("Epoch:", epoch, "w=", w, "loss = ", loss_val)

plt.plot(mse_list)
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.title("loss in each epoch")
plt.show()


