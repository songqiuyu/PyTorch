"""
    Advanced CNN
    GoogleNet
"""

# 减少代码冗余
"""
1*1卷积的主要作用就是为了升维或者降维
这样就会将前面N通道的数值做一个加法运算，实现降维

张量的维度是(batch, channel, width, height)
"""
"""
实现Google Net
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


batch_size = 32

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
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

test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size,
                         num_workers=2)

# GoogleNet的一个子模块, 即InceptionA
class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branchx1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branchx1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        # 这一步就是Concatenate
        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        # 在channel通道上合并 (b, c, w, h) c为 dim = 1
        return torch.cat(outputs, dim=1)


# 完整网络由多个子模块InceptionA组成，这里实现两个
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 刚开始图像是(b, 1, w, h) 十个卷积核 (b, 10, w-4, h-4)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)
        # 定义两个子模块
        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)
        # 定义池化层
        self.mp = nn.MaxPool2d(2)
        # 分类是1408个维度，分成最后10维
        self.fc = nn.Linear(1408, 10)

    def forward(self, x):
        # size(0)表示的是一个batch中的个数，即b
        in_size = x.size(0)
        # 先conv1卷积池化激活函数, C为10
        x = F.relu(self.mp(self.conv1(x)))
        # 然后输入incep1
        x = self.incep1(x)
        # 然后conv2卷积池化激活函数, C为88
        x = F.relu(self.mp(self.conv2(x)))
        # 然后输入incep2
        x = self.incep2(x)
        # 在输入fc的时候，要做一个Flatten，我们让其自动根据batch数目得到
        x = x.view(in_size, -1) #x.shape = (b, 1408)
        x = self.fc(x)
        return x

# 开始使用GoogleNet训练
model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        #1.先优化器清零梯度
        optimizer.zero_grad()
        #然后前馈计算
        outputs = model(inputs)
        #计算loss
        loss = criterion(outputs, labels)
        #反向传播
        loss.backward()
        #step下一步
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print("[%d, %5d] loss: %.3f" % (epoch+1, batch_idx+1, running_loss / 300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # 返回的是最大值和最大值的索引，我们只需要索引，不需要值，值保存在_中
            # 维度从1开始是因为，这是一个batch的结果,dim0为batch
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy on test set: %d %%" % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()





