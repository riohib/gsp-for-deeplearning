# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
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

from models.vgg import VGG
from gsp_methods import *

import sys 
sys.path.append('/data/users2/rohib/github/testing')
import utils_gsp.sps_tools as sps_tools
from torch.utils.tensorboard import SummaryWriter


# %%
class Args:
    lr = 0.1
    resume = False
    gsp_sps = 0.8
    gsp_int = 3
    start_epoch = -1

args = Args

writer = SummaryWriter()


# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# %%
# Model
print('==> Building model..')
# net = VGG(depth=16, dataset='cifar10', batchnorm=True)
model = VGG('VGG19')
model = model.to(device)
# model


# %%
bind_gsp_methods_to_model(model, args, apply_gsp=True)


# %%
ct = 0
for name, param in model.named_parameters():
    # print(name)
    ct+=1
print(ct)

# %%
chk_name = 'gsp_block'
args.resume = True
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/{chk_name}.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# %%
len(optimizer.param_groups[0]['params'])


# %%
best_acc
def gsp_sparse_training(model, args):
    # Additional Class Variables for GSP
    print(f"ARGS GSP INT: {args.gsp_int}")
    model.sps = args.gsp_sps
    model.curr_iter = 0
    model.start_gsp_epoch = -1
    model.gsp_int = args.gsp_int
    model.logger = None
    model.gsp_training_mode = True

    if args.resume:
        model.curr_epoch = args.start_epoch
        print(f"Current Epoch: {args.start_epoch}")
    else:
        model.curr_epoch = 0


# %%
print(best_acc)


# %%
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 300 == 0:
            print( f"[{batch_idx}/{len(trainloader)}], Loss: {(train_loss/(batch_idx+1))} | Acc: {100.*correct/total} " )


# %%
def test(model, criterion, testloader, epoch, args):
    global best_acc
    model.eval()
    model.gsp_training_mode = False

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    print(f"Accuracy: {acc}")

test(model, criterion, testloader, 1, args)


# %%
def gsp_sparse_training(model, args):
    # Additional Class Variables for GSP
    model.sps = args.gsp_sps
    model.curr_iter = 0
    model.start_gsp_epoch = 0
    model.gsp_int = args.gsp_int
    model.logger = None

    if args.resume:
        model.curr_epoch = args.start_epoch
        print(f"Current Epoch: {args.start_epoch}")
    else:
        model.curr_epoch = 0


# %%
# model = net
model.curr_epoch = 1
model.curr_iter = 0
model.gsp_training_mode = True


# %%
gsp_sparse_training(model, args)
# bind_gsp_methods_to_model(model, args, apply_gsp=True)


# %%
for name, layer in model.named_modules():
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        print(layer.gsp_w.shape)


# %%
# net.initialize_gsp_layers = initialize_gsp_layers.__get__(net)
# net.initialize_gsp_layers()


# %%
# net.features[0].gsp_w
# bind_gsp_methods_to_model(net, args)
# net.logger = 1

# %% [markdown]
# ### Pass Minibatch

# %%
print(f"epoch: {model.curr_epoch} | iter: {model.curr_iter}")

images, target = next(iter(trainloader))

images = images.cuda('cuda:0', non_blocking=True)
target = target.cuda('cuda:0', non_blocking=True)

optimizer.zero_grad()
output = model(images)
loss = criterion(output, target)

model.curr_iter += 1


# %%
# print(model.features[0].weight.grad)
print(model.features[7].gsp_gate)


# %%
loss.backward()
optimizer.step()


# %%
torch.norm(model.features[7].gsp_gate.grad)


# %%
# sps_tools.get_layerwise_sps(net)
# sps_tools.get_abs_sps(net)a

images, target = next(iter(trainloader))
images = images.cuda('cuda:0', non_blocking=True)
target = target.cuda('cuda:0', non_blocking=True)

# writer.add_graph(model, images)


# %%
model.features(images).shape


# %%
for name, layer in model.named_modules():
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        layer 
        break


# %%
layer.hook


# %%
str(layer.weight.device)


# %%



