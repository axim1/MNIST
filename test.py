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


model = torch.load('model.pth')
for i in range(5):
    exm=next(iter(testloader))
    #batchnum, (exmdata,exmtar)=next(exm)

    images, labels = exm
    preds = net(images)
    fig2=plt.figure()
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(images[0][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(labels.item()))
    plt.xticks([])
    plt.yticks([])
    fig2
    print (labels.item())