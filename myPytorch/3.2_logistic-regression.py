
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(2017)

# Get data - Numpy
with open('./data.txt', 'r') as f:
    data_list = [i.split('\n')[0].split(',') for i in f.readlines()]
    data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]

x0_max = max([i[0] for i in data])
x1_max = max([i[1] for i in data])
data = [(i[0]/x0_max, i[1]/x1_max, i[2]) for i in data]

x0 = list(filter(lambda x: x[-1] == 0.0, data)) 
x1 = list(filter(lambda x: x[-1] == 1.0, data)) 

get_ipython().run_line_magic('matplotlib', 'inline')
plot_x0 = [i[0] for i in x0]
plot_y0 = [i[1] for i in x0]
plot_x1 = [i[0] for i in x1]
plot_y1 = [i[1] for i in x1]
plt.plot(plot_x0, plot_y0, 'ro', label='x_0')
plt.plot(plot_x1, plot_y1, 'bo', label='x_1')
plt.legend(loc='best')

'''
# Sigmoid, Logistic regression
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

plot_x = np.arange(-10, 10.01, 0.01)
plot_y = sigmoid(plot_x)
plt.plot(plot_x, plot_y, 'r')

'''

# Inital Cutting Line
w = Variable(torch.randn(2, 1), requires_grad=True) 
b = Variable(torch.zeros(1), requires_grad=True)
w0 = w[0].data[0].numpy()
w1 = w[1].data[0].numpy()
b0 = b.data[0].numpy()

plot_x = np.arange(0.2, 1, 0.01)
plot_y = (-w0 * plot_x - b0) / w1
plt.plot(plot_x, plot_y, 'g', label='cutting line')
plt.plot(plot_x0, plot_y0, 'ro', label='x_0')
plt.plot(plot_x1, plot_y1, 'bo', label='x_1')
plt.legend(loc='best')


# Logistic_regression
import torch.nn.functional as F
def logistic_regression(x):
    return F.sigmoid(torch.mm(x, w) + b)

def binary_loss(y_pred, y):
    logits = (y * y_pred.clamp(1e-12).log() + (1 - y) * (1 - y_pred).clamp(1e-12).log()).mean()
    return -logits

np_data = np.array(data, dtype='float32')
x_data = torch.from_numpy(np_data[:, 0:2]) 
y_data = torch.from_numpy(np_data[:, -1]).unsqueeze(1) 
x_data = Variable(x_data)
y_data = Variable(y_data)
y_pred = logistic_regression(x_data)
loss = binary_loss(y_pred, y_data)
print(loss)

'''
loss.backward()
w.data = w.data - 0.1 * w.grad.data
b.data = b.data - 0.1 * b.grad.data
y_pred = logistic_regression(x_data)
loss = binary_loss(y_pred, y_data)
print(loss)
'''

# update with torch.optim 
from torch import nn
w = nn.Parameter(torch.randn(2, 1))
b = nn.Parameter(torch.zeros(1))

def logistic_regression(x):
    return F.sigmoid(torch.mm(x, w) + b)
optimizer = torch.optim.SGD([w, b], lr=1.)

import time
start = time.time()
for e in range(1000):
    # forward
    y_pred = logistic_regression(x_data)
    loss = binary_loss(y_pred, y_data) 
    # backward
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step() 
    # accuracy
    mask = y_pred.ge(0.5).float()
    acc = (mask == y_data).sum().data[0] / y_data.shape[0]
    if (e + 1) % 200 == 0:
        print('epoch: {}, Loss: {:.5f}, Acc: {:.5f}'.format(e+1, loss.data[0], acc))
during = time.time() - start
print()
print('During Time: {:.3f} s'.format(during))

# weights after 1000 update using ...torch.optim.SGD([w,b], lr=1)
w0 = w[0].data[0].numpy()
w1 = w[1].data[0].numpy()
b0 = b.data[0].numpy()

plot_x = np.arange(0.2, 1, 0.01)
plot_y = (-w0 * plot_x - b0) / w1

plt.plot(plot_x, plot_y, 'g', label='cutting line')
plt.plot(plot_x0, plot_y0, 'ro', label='x_0')
plt.plot(plot_x1, plot_y1, 'bo', label='x_1')
plt.legend(loc='best')


# loss implemented by pytorch ================================================
'''
nn.MSE()   # linear regression
nn.BCEWithLogitsLoss()  # logistic regression
more loss functions: https://pytorch.org/docs/0.3.0/nn.html#loss-functions
'''

criterion = nn.BCEWithLogitsLoss() # sigmoid 和 loss 写在一层，有更快的速度、稳定性
w = nn.Parameter(torch.randn(2, 1))
b = nn.Parameter(torch.zeros(1))

def logistic_reg(x):
    return torch.mm(x, w) + b

optimizer = torch.optim.SGD([w, b], 1.)
y_pred = logistic_reg(x_data)
loss = criterion(y_pred, y_data)
print(loss.data)

start = time.time()
for e in range(1000):
    # forward
    y_pred = logistic_reg(x_data)
    loss = criterion(y_pred, y_data)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # accuracy
    mask = y_pred.ge(0.5).float()
    acc = (mask == y_data).sum().data[0] / y_data.shape[0]
    if (e + 1) % 200 == 0:
        print('epoch: {}, Loss: {:.5f}, Acc: {:.5f}'.format(e+1, loss.data[0], acc))

during = time.time() - start
print()
print('During Time: {:.3f} s'.format(during))
