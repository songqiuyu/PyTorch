"""
    1. Convolution Layer
    2. Subsampling  => MaxPooling Layer
    3. Fully Connected Layer
    So, CNN = Feature Extract + Classification

    Filter is 4-Dimension
"""
"""
import torch

in_channels, out_channels = 5, 10
width, height = 100, 100
kernel_size = 3
batch_size = 1

input = torch.randn(batch_size,
                    in_channels,
                    width,
                    height)

conv_layer = torch.nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size=kernel_size)

output = conv_layer(input)

print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)

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

batch_size = 256

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
                          batch_size=batch_size,
                          num_workers=2)

test_dataset = torchvision.datasets.MNIST(root='./data/mnist/',
                                          train=False,
                                          download=True,
                                          transform=transform)
# 测试集一般不用shuffle
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size,
                         num_workers=2)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Conv2d(input_channels, output_channels,kernel_size, padding, stride)
        # (batch, 1, 28, 28)
        self.c1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        # (batch, 10, 24, 24)
        # self.p1 = torch.nn.MaxPool2d(kernel_size=2)
        # (batch, 10, 12, 12)
        self.c2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        # (batch, 20, 8, 8)
        self.pooling = torch.nn.MaxPool2d(kernel_size=2)
        # (batch, 20, 4, 4) = (batch, 320)  可以自行计算，也可以用Flatten
        self.l1 = torch.nn.Linear(320, 256)
        self.l2 = torch.nn.Linear(256, 128)
        self.l3 = torch.nn.Linear(128, 64)
        self.l4 = torch.nn.Linear(64, 10)

        """
            因为池化操作没变，所以定义一个self.pooling层就行了，不用定义两个
            调用的时候调用两次
        """


    def forward(self, x):
        batch_size = x.size(0)
        # input (batch, 1, 28, 28)
        x = self.c1(x)
        x = F.relu(self.pooling(x))
        x = self.c2(x)
        x = F.relu(self.pooling(x))
        # input (batch, 320)
        x = x.view(batch_size, -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x

model = Net()
# 使用显卡,cuda:0第一块显卡
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 将模型迁移到GPU上
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
# 训练需要backward
def train(epoch):
    # 运行的损失
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, labels = data
        # 将运算数据也迁移到显卡上
        inputs, labels = inputs.to(device), labels.to(device)
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
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # outputs为(N * 10),我们从第一维看，输出每一个的最大值
            _, predicted  = torch.max(outputs.data, dim=1 )
            # 去labels的第0个元素，即N，因为labels是(N * 10)
            total += labels.size(0)
            # 因为labels和predicted都是张量，因此用item
            correct += (predicted == labels).sum().item()
    print("Accuracy on test set: %d %%" % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
