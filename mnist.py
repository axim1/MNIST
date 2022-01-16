#%%
from os import path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision.datasets.utils import download_and_extract_archive
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
#from plotcm import plot_confusion_matrix

import pdb

torch.set_printoptions(linewidth=120)

trainset= torchvision.datasets.MNIST(
    root='./data'
    ,train=True
    ,download= True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

trainloader=torch.utils.data.DataLoader(trainset,batch_size=1000,shuffle=True)
testset=torchvision.datasets.MNIST(
    root='./data',train=False,download=True
    ,transform=transforms.Compose([transforms.ToTensor()])

)
testloader=torch.utils.data.DataLoader(testset,batch_size=1,shuffle=True)
print(trainset.targets.bincount())




class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5)
        self.fc1 = nn.Linear(12*4*4, 120)
        self.fc2 = nn.Linear(120, 60)
        self.out=nn.Linear(60,10)
    def forward(self,x):
        x=self.conv1(x)
        x=F.max_pool2d(x, kernel_size=2, stride=2)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.max_pool2d(x, kernel_size=2, stride=2)
        x=F.relu(x)
                # (4) hidden linear layer
        t = x.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)
                # (6) output layer
        t = self.out(t)
        t = F.softmax(t, dim=1)
        return t
#%%
net=Net()
optimizer=optim.Adam(net.parameters(), lr=0.01)



# %%
total_loss=0
total_correct=0
epoch=0
batch = next(iter(trainloader))
for epoch in range(5):
    for batch in trainloader:
        images, labels = batch
        preds = net(images)
        loss = F.cross_entropy(preds, labels)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
        total_correct += preds.argmax(dim=1).eq(labels).sum().item()
    print(
    "epoch:", 0, 
    "total_correct:", total_correct, 
    "loss:", total_loss)
# %%
torch.save(net, "model.pth")
# %%
fig2=plt.figure()
for i in range(15):
    exm=next(iter(testloader))
    #batchnum, (exmdata,exmtar)=next(exm)

    images, labels = exm
    preds = net(images)
    
    plt.subplot(5,3,i+1)
    plt.tight_layout()
    plt.imshow(images[0][0], cmap='gray', interpolation='none')
    plt.title("Prediction : {}".format(preds.argmax().item())) #+"Ground Truth: {}".format(labels.item()))
    plt.xticks([])
    plt.yticks([])
    
    print ()

fig2

# %%
