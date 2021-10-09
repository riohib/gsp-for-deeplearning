'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from models.vgg import VGG
from models.vgg_sps import SUBVGG
from gsp_methods import *
import statistics

# import utils.ipynb_funcs as utilfuncs

import math
import sys 
sys.path.append('/data/users2/rohib/github/testing')
import utils_gsp.sps_tools as sps_tools
import utils_gsp.padded_gsp as gsp_pad

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt


def network_bar_plot(model):
    plt.style.use('seaborn')
    fig, ax = plt.subplots(nrows=4, ncols=5)

    shapes_l = list()
    gates_l = list()
    gates_norm = list()
    gate_cat = np.array([])
    gate_cat_torch = torch.tensor([]).cuda()


    count = 0
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):

            # data = np.abs(layer.gsp_gate.detach().clone().flatten().cpu().numpy())
            data_torch = torch.abs(layer.gsp_gate.detach().clone().flatten())
            # data = data_torch.cpu().numpy()

            # shapes_l.append(data_torch.shape[0])
            # dmin = data_torch.min(); dmax= data_torch.max()
            # data_torch -= dmin
            # data_torch /= (dmax - dmin)

            data = data_torch.cpu().numpy()
            gates_l.append(data)
            gates_norm.append(data)

            # gate_cat = np.concatenate((gate_cat, data))
            # if 'classifier' not in name:
            gate_cat_torch = torch.cat( (gate_cat_torch, data_torch) )

            x_vals = list(range(data.shape[0]))
            ax.reshape(-1)[count].bar( x_vals , data)
            ax.reshape(-1)[count].set_title(name)
            count += 1

    fig.set_size_inches(17, 16)
    fig.suptitle("Trained Gate Magnitude (y-axis) corresponding to each filter (x-axis) of VGG-16." \
                "One subplot per layer!", fontsize=18)
    plt.savefig('gate_bar.png')