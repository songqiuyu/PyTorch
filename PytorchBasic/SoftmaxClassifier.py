"""
    softmax
        1.使用exp()
        2.使用onehot为结果

    满足概率大于0
    满足项间抑制
    使用exp()则必定是正的
    使用exp(x) / exp(x) + exp(y) + exp(z)实现抑制且总和为1
    使用onehot,这样损失函数除了y_hat其余项都是0，就成了-lny_hat的情况！

    如果使用torch.nn.CrossEntropyLoss()
    其中包含了Softmax, Log 和 One-hot因此不需要输出Softmax层
    即运算到最后一层即可

    torch.nn.NLLLoss
    CrossEntropyLoss <==> LogSoftmax + NLLLoss
"""

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

batch_size = 64

# Convert the PIL Image to Tensor and Normalize to 0-1
# PIL Image Z(28, 28) [0..255]  -> R(1, 28, 28) [0..1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data/mnist/',
                                           train=True,
                                           download=True,
                                           transform=transform)

train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

test_dataset = torchvision.datasets.MNIST(root='./data/mnist/',
                                          train=False,
                                          download=True,
                                          transform=transform)
# 测试集一般不用shuffle
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 其中N表示一个批次里面的图片个数
        # (N, 784) -> (N, 512)
        self.l1 = torch.nn.Linear(28 * 28, 512)
        # (N, 512) -> (N, 256)
        self.l2 = torch.nn.Linear(512, 256)
        # (N, 256) -> (N, 128)
        self.l3 = torch.nn.Linear(256, 128)
        # (N, 128) -> (N, 64)
        self.l4 = torch.nn.Linear(128, 64)
        # (N, 64) -> (N, 10)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        # -1就是自动获取批次的N
        x = x.view(-1, 28 * 28)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = self.l5(x)
        return x


model = Net()
# CrossEntropyLoss <==> SoftmaxLog + NLLLoss
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# 训练需要backward
def train(epoch):
    # 运行的损失
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, labels = data
        # zero is need
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print("[%d, %5d] loss: %.3f" % (epoch+1, batch_idx+1, running_loss / 300))
            running_loss = 0.0

# 测试不需要backward
def test():
    correct = 0
    total = 0
    # 不计算梯度
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            # outputs为(N * 10),我们从第一维看，输出每一个的最大值
            # 返回的是最大值和最大值的索引，我们只需要索引，不需要值，值保存在_中
            _, predicted  = torch.max(outputs.data, dim=1)
            # 去labels的第0个元素，即N，因为labels是(N * 10)
            total += labels.size(0)
            # 因为labels和predicted都是张量，因此用item
            correct += (predicted == labels).sum().item()
    print("Accuracy on test set: %d %%" % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()


