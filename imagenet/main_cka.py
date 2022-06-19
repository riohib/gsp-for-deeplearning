import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import copy

from datetime import datetime

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))



import sys
sys.path.append('./models')


from main_model import *
import sys 
sys.path.append('/data/users2/rohib/github/testing')
import utils_gsp.sps_tools as sps_tools
import utils_gsp.gpu_projection as gsp_gpu

import sys
sys.path.append('./models')
import models.resnet_torch as ResNet


from models.finetuners import *
from apply_gsp import GSP_Model

class Args:
    data = '/datasets01/imagenet_full_size/061417/'
    arch = 'resnet50'
    workers = 4
    epochs = 1
    start_epoch = 0
    batch_size = 16
    lr = 0.1
    momentum = 0.9
    weight_decay = 1e-4
    print_freq = 10
    resume = "/private/home/riohib/explore/gsp-for-deeplearning/imagenet/results/gsp_S80_fts90/mask_stripped_best.pth.tar"
    evaluate = False
    pretrained = False
    world_size = -1
    dist_url = 'tcp://224.66.41.62:23456'
    dist_backend = 'nccl'
    seed = None
    gpu = None
    multiprocessing_distributed = False

args = Args