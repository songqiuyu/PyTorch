import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class TitanicDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class ClassificationModel(torch.nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()

        self.linear1 = torch.nn.Linear(6, 32)
        self.linear2 = torch.nn.Linear(32, 16)
        self.linear3 = torch.nn.Linear(16, 8)
        self.linear4 = torch.nn.Linear(8, 1)

        self.Relu = torch.nn.ReLU()
        self.Sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.Relu(self.linear1(x))
        x = self.Relu(self.linear2(x))
        x = self.Relu(self.linear3(x))
        x = self.Relu(self.linear4(x))
        return x

dataset = TitanicDataset('./data/Titanic/train.csv')
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)
model = ClassificationModel()
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

bce_list = []

if __name__ == '__main__':
    for epoch in range(10):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            bce_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()


    plt.plot(bce_list)
    plt.xlabel("n")
    plt.ylabel("loss")
    plt.title("loss in training set")
    plt.show()