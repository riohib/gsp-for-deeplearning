
# %%
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

from datetime import datetime

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


# %%
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


# %%
# model = models.__dict__['resnet18'](pretrained=False)


# %%
class Args:
    data = '/data/users2/rohib/github/imagenet-data'
    arch = 'resnet18'
    workers = 4
    epochs = 1
    start_epoch = 0
    batch_size = 16
    lr = 0.1
    momentum = 0.9
    weight_decay = 1e-4
    print_freq = 10
    resume = ''
    evaluate = False
    pretrained = False
    world_size = -1
    dist_url = 'tcp://224.66.41.62:23456'
    dist_backend = 'nccl'
    seed = None
    gpu = None
    multiprocessing_distributed = False

args = Args


# %%
args.multiprocessing_distributed

# %%
def get_model(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                        'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        model, train_loader, optimizer, criterion = main_worker(args.gpu, ngpus_per_node, args)
    return model, train_loader, optimizer, criterion

model, train_loader, optimizer, criterion = get_model(args)


# %%
gsp_func = gsp_gpu
sps = 0.8


# %%
optimizer.param_groups[0]['lr']


# %%
print(model.module.curr_epoch)
model.module.curr_epoch = 1
model.module.curr_iter = 0

model.module.gsp_int = 2
model.module.sps = 0.9
model.module.gsp_training_mode=True


# %%
print(model.module.curr_epoch)
print(model.module.curr_iter)
print(model.module.gsp_int)
print(model.module.gsp_training_mode)
print(model.module.sps)


# %%
print(f'Current Epoch: {model.module.curr_epoch}')
print(f'Current iter: {model.module.curr_iter}')

images, target = next(iter(train_loader))
images = images.cuda(args.gpu, non_blocking=True)
target = target.cuda(args.gpu, non_blocking=True)

output = model(images)
model.module.curr_iter += 1


# %%
print(sps_tools.get_abs_sps(model.module)[0])

# model.module.conv1

sps_tools.get_layerwise_sps(model)


# %%
weight = model.module.conv1.weight.data
gsp_in = weight.reshape( weight.shape[0], -1)

sps_tools.sparsity(gsp_in)
gsp_func.groupedsparseproj


# %%
# compute output
loss = criterion(output, target)

# measure accuracy and record loss
acc1, acc5 = accuracy(output, target, topk=(1, 5))
losses.update(loss.item(), images.size(0))
top1.update(acc1[0], images.size(0))
top5.update(acc5[0], images.size(0))

# compute gradient and do SGD step
optimizer.zero_grad()
loss.backward()
optimizer.step()


# %%
# sps_tools.get_abs_sps(model)
# model, train_loader = get_model(args)
masks = get_conv_linear_mask(model.module)
print(masks.keys())


# %%
bind_gsp_methods_to_model(model.module)


# %%
masks.keys()


# %%
# model.module.register_mask(masks)

for name, layer in model.module.named_modules():
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        # assert (layer.weight.data.shape == in_masks[name]), f"Weight and mask shape mismatch in layer: {self}"
        print(name)


# %%
count =0
orig_shape={}
layer_d = {}
gsp_in_d = {}
post_gsp_shp = {}

for name, layer in model.named_modules():
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        layer_d[name] = layer
        orig_shape[name] = layer.weight.shape
        w_shape = layer.weight.shape
        
        if 'downsample' in name:
            # print(layer.weight.shape)
            dim_1 ,  dim_2 = layer.weight.shape[0], layer.weight.shape[1]
            gsp_in_d[name] = layer.weight.detach().reshape(dim_1, -1)
        else:
            # print(layer.weight.shape[0])
            gsp_in_d[name] = layer.weight.detach().reshape(layer.weight.shape[0], -1)
        
        # print(gsp_in_d[name].shape)
        # sparse_l = gsp_gpu.groupedsparseproj(gsp_in_d[name].T, sps=0.8).T
        # print(sparse_l.shape)
        layer.weight.data = gsp_gpu.groupedsparseproj(gsp_in_d[name].T, sps=0.8).T.reshape(w_shape)
        post_gsp_shp[name] = layer.weight.shape
        
# gsp_shape_d = [x.shape for x in gsp_in_d.values()]
# list(gsp_shape_d)


# %%
# gsp_shape_d = [x.shape for x in gsp_in_d.values()]
# list(post_gsp_shp.values())


# %%
print(layer_d['module.layer2.0.downsample.0'].weight.shape)
print(layer_d['module.layer3.0.downsample.0'].weight.reshape(128, -1).shape)
# list(gsp_in_d.keys())
# gsp_shape_d = [x.shape for x in gsp_in_d.values()]

# list(gsp_shape_d)


# %%
print(layer.weight.shape)
sparse_weight.shape


# %%



# %%
print(f"The total model sparsity is: {sps_tools.get_abs_sps(model)[0].item()}")
sps_tools.gsp_conv_linear(model, sps=0.9)
print(f"The total model sparsity after GSP: {sps_tools.get_abs_sps(model)[0].item()}")


# %%



# %%
model_test = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )


# %%
for name, param in model_test.named_parameters():
    print(name)


# %%



