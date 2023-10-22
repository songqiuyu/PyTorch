"""
    DeepLearning Procedure
    1. DataSet:
        Divide Dataset into Training Set, Validation Set and Testing Set
    2. Model
        We Define Two PART
        Forward()
        Loss()
    3. Training
    4. Inferring
"""

"""
    Linear Model  
    y_hat = x * w + b
    Loss Function
    loss = Mean Square Error (MSE)
    optimization the loss (value approximate -> 0)
"""

import numpy as np  # Numpy
import matplotlib.pyplot as plt # Chart

# Training Set
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# forward: Linear Function
def forward(x):
    return x * w

# Define the Loss Function
def loss(x, y):
    # the predicted value of y
    y_pred = forward(x)
    return (y - y_pred) * (y - y_pred)

w_list = []     # value of w
mse_list = []   # MSE Mean Square Error
# np.arange是值的序列
for w in np.arange(0.0, 4.1, 0.1):
    print("w", w)
    l_sum = 0   # the sum of Loss value
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print("MSE=", l_sum / len(x_data))
    w_list.append(w)
    mse_list.append(l_sum / len(x_data))

plt.plot(w_list, mse_list)
plt.ylabel('loss')
plt.xlabel('w')
plt.title('w value corresponds to loss value')
plt.show()








