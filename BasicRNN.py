"""
    RNN 循环神经网络
    RNN的数据为(seqLen, batchSize, inputSize)


    input shape = (batchSize, inputSize)
    output shape = (batchSize, hiddenSize)
    dataset shape = (seqLen, batchSize, inputSize)
    使用torch.nn.RNNCell(input_size, hidden_size)构建RNNCell

    用RNN一定要把维度搞清楚，RNN的数据维度是
    (seqLen, batchSize, inputSize)
"""

"""
target: seq2seq
hello -> ohlol
"""
import torch


batch_size = 1
# 独热编码4
input_size = 4
# 对应四个字母的概率
hidden_size = 4


idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]
# 独热编码对应0 1 2 3 -> e h l o
one_hot_lookup = [[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]]

# 将x_data转化成独热编码
x_one_hot = [one_hot_lookup[x] for x in x_data]



# seqLen我们自动生成，batch_size和input_size我们定
inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
# labels第一位还是seqLen自动生成，那么labels标签就是一个vector，1维
labels = torch.LongTensor(y_data).view(-1, 1)

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 一个RnnCell，即下方的输入vector，上方的输出vector
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size,
                                        hidden_size=self.hidden_size)
    # 每一层都要有一个输入和上一层输出
    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return hidden

    # 初始化h0,h0还没算出来
    def init_hidden(self):
        #h0不需要管seqLen，因为就他自己，h0，大小就是batch_size和hidden_size
        return torch.zeros(self.batch_size, self.hidden_size)

net = Model(input_size, hidden_size, batch_size)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

for epoch in range(15):
    loss = 0

    # 梯度清零
    optimizer.zero_grad()
    # 初始化h0
    hidden = net.init_hidden()
    print("Predicted string:", end='')
    # inputs.shape = (seqLen, batchSize, inputSize)
    # input.shape = (batchSize, inputSize)
    # 实际上input是每次循环拿序列中的第i个，所以loss的反向传播也是最后循环结束才计算
    for input, label in zip(inputs, labels):
        hidden = net(input, hidden)
        # 这里loss是构建计算图，即每一个input的loss都要构建一个Node最后加起来
        loss += criterion(hidden, label)
        _, idx = hidden.max(dim=1)
        print(idx2char[idx.item()], end='')
    # 反向传播
    loss.backward()
    optimizer.step()
    print(", Epoch [%d/15] loss=%.4f" % (epoch+1, loss.item()))

