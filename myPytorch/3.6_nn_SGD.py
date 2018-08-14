
# coding: utf-8

# In[2]:


import numpy as np
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def data_tf(x):
    x = np.array(x, dtype='float32') / 255 
    x = (x - 0.5) / 0.5 
    x = x.reshape((-1,)) #
    x = torch.from_numpy(x)
    return x

train_set = MNIST('./data', train=True, transform=data_tf, download=True)
test_set = MNIST('./data', train=False, transform=data_tf, download=True)
criterion = nn.CrossEntropyLoss()

#optimzier = torch.optim.SGD(net.parameters(), 1e-2)
def sgd_update(parameters, lr):
    for param in parameters:
        param.data = param.data - lr * param.grad.data


# In[12]:


train_data = DataLoader(train_set, batch_size = 1, shuffle = True)

net = nn.Sequential(
    nn.Linear(784, 200),
    nn.ReLU(),
    nn.Linear(200,10),
)


# In[14]:


# ================= batch_size = 1, losses =====================

losses1 = []
idx = 0

start = time.time()
for e in range(5):
    train_loss = 0
    
    for im, label in train_data:
        im = Variable(im)
        label = Variable(label)
        # forward
        out = net(im)
        loss = criterion(out, label )
        # backward
        net.zero_grad()
        loss.backward()
        sgd_update(net.parameters(), 1e-2)
        # loss
        train_loss += loss.data[0]
        if idx % 30 == 0:
            losses1.append(loss.data[0])
        idx += 1
    print('epoch: {}, Train Loss: {:.6f}'.format(e, train_loss / len(train_data)))

    end = time.time() # 计时结束
print('使用时间: {:.5f} s'.format(end - start))    


# In[31]:


x_axis = np.linspace(0,10, len(losses1), endpoint=True)
plt.semilogy(x_axis, losses1, label='batch_size=1')
plt.legend(loc='best')


# In[33]:


# ================= batch_size = 64, losses ===================== w
train_data = DataLoader(train_set, batch_size=64, shuffle=True)

net = nn.Sequential(
    nn.Linear(784, 200),
    nn.ReLU(),
    nn.Linear(200,10)
)

losses2 = []
idx = 0
start = time.time()
for e in range(5):
    train_loss = 0
    
    for im, label in train_data:
        im = Variable(im)
        label = Variable(label)
        # forward
        out = net(im)
        loss = criterion(out, label)
        # backward
        net.zero_grad()
        loss.backward()
        sgd_update(net.parameters(),1e-2)
        # error
        train_loss += loss.data[0]
        if idx % 30 == 0:
            losses2.append(loss.data[0])
        idx += 1
    print('epoch:{},Train Loss: {:.6f}'.format(e, train_loss / len(train_data)))
    
end = time.time()
print('time: {:.5f} s'.format(end - start))
        


# In[39]:


# - Why Batch - 
# batch = 64, training process faster, flutuate less
# ----

x_axis =  np.linspace(0, 5, len(losses2), endpoint=True)
plt.semilogy(x_axis, losses2, label='batch_size=64')
plt.legend(loc='best')


# In[4]:


# ================= increase learning rate ===================== 
train_data = DataLoader(train_set, batch_size = 64, shuffle=True)

net = nn.Sequential(
    nn.Linear(784, 200),
    nn.ReLU(),
    nn.Linear(200, 10),
)

losses3 = []
idx = 0
start = time.time()

for e in range(5):
    train_loss = 0
    
    for im, label in train_data:
        im = Variable(im)
        label = Variable(label)
        # forward
        out = net(im)
        loss = criterion(out, label)
        # backward
        net.zero_grad()
        loss.backward()
        sgd_update(net.parameters(),1) #optimzier = torch.optim.SGD(net.parameters(), 1e-2)
        
        # loss
        train_loss += loss.data[0]
        if idx % 30 ==0:
            losses3.append(loss.data[0])
        idx += 1
    print('epoch: {}, Train Loss: {:.6f}'
          .format(e, train_loss / len(train_data)))
    
end = time.time()
print('time: {:.5f} s'.format(end - start))



# In[5]:


x_axis = np.linspace(0, 5, len(losses3), endpoint=True)
plt.semilogy(x_axis, losses3, label='lr = 1')
plt.legend(loc='best')


# In[6]:



train_datatrain_d  = DataLoader(train_set, batch_size=64, shuffle=True)
net = nn.Sequential(
    nn.Linear(784, 200),
    nn.ReLU(),
    nn.Linear(200, 10),
)

optimzier = torch.optim.SGD(net.parameters(), 1e-2)
start = time.time() 
for e in range(5):
    train_loss = 0
    for im, label in train_data:
        im = Variable(im)
        label = Variable(label)
        # forward
        out = net(im)
        loss = criterion(out, label)
        # backward
        optimzier.zero_grad()
        loss.backward()
        optimzier.step()
        # loss
        train_loss += loss.data[0]
    print('epoch: {}, Train Loss: {:.6f}'
          .format(e, train_loss / len(train_data)))
    
end = time.time() 
print('time: {:.5f} s'.format(end - start))

