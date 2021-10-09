'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import torch
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statistics

# import utils.ipynb_funcs as utilfuncs


def plot_weight_dist(model, title=None, arch='resnet20', filename='./w_dist.png'):
    
    cfg = {'resnet20':(4,5)}
    title = "Weight Distribution of the Model!" if (title == None) else title
    
    plt.style.use('seaborn')
    fig, ax = plt.subplots(nrows=cfg[arch][0], ncols=cfg[arch][1])
    
    count = 0
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):

            weights = layer.weight.data.detach()
            in_data = torch.histc(weights, bins=100).cpu().numpy()

            x_vals = np.arange(len(in_data))

            ax.reshape(-1)[count].bar( x_vals, in_data)
            ax.reshape(-1)[count].set_title(name)
            count += 1

    fig.set_size_inches(17, 16)
    fig.suptitle(title, fontsize=18)
    plt.savefig(filename)