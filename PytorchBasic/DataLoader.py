"""
for epoch in range(train_epochs):
        pass
    for i in range(total_batch):
        pass

Dataloader(batch_size=, shuffle=)
"""
import torch
import numpy as np
# Data需要我们自定义类去继承
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        """
            1. 将整个数据读入内存 All Data
            2. 如果数据特别大，将文件路径读入，现用现读
        """
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])

    # 拿出数据集的某个数据
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    # 输出数据集的长度
    def __len__(self):
        return self.len

dataset = DiabetesDataset('')
# num_workers是用几个线程
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)

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

if __name__ == '__main__':
    # 训练多少轮
    for epoch in range(100):
        # train_loader中拿出来的是x和y，0表示从索引0开始
        for i, data in enumerate(train_loader, 0):
            # step1 Prepare data: get x, y
            inputs, labels = data
            # step2 Forward
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            # step3 Backward
            optimizer.zero_grad()
            loss.backward()
            # step4 Update
            optimizer.step()
