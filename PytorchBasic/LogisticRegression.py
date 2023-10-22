"""
    激活函数的功能仅仅就是将实数域的值映射到0-1上去
    至于wx+b的输入对应概率是多少，是让神经网络自己去算的
    注意，激活函数的功能之一就是数据归一化（映射）
    
    loss = -(ylog(y_hat) + (1 - y)log(1 - y_hat))
    如果真实值是0
        则loss = -log(1-y_hat)
            log是增函数，-log是减函数，要想loss最小则y_hat无限趋近于0最好
    如果真实值是1
        则loss = -log(y_hat)
            则y_hat越趋近于1，loss越小
    这就是交叉熵损失函数cross_entropy
        BCE Loss
"""

import torch.nn.functional as F
import torch

class LogicalRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogicalRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # 在前馈完成线性运算后调用sigmoid函数对结果进行非线性化
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])

model = LogicalRegressionModel()
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print("y_pre =", y_test.item())



