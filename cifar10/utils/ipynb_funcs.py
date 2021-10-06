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

best_acc = 0

def get_model_layers(model):
    names = list()
    weight_l = list()
    shape_l = list()
    weight_d = dict()
    layers = dict()
    gates_d = dict()
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weight_d[name] = layer.weight
            weight_l.append(layer.weight)
            shape_l.append(layer.weight.shape)
            layers[name] = layer
            names.append(name)
            
    return names, weight_l, shape_l, weight_d, layers


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



# Training
def train(model, criterion, testloader, epoch, args, mode=True):
    model.gsp_training_mode = mode
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
        
        if batch_idx % 100 == 0:
            # print( f"[{batch_idx}/{len(trainloader)}], Loss: {(train_loss/(batch_idx+1))} | Acc: {100.*correct/total} " \
            #        f"grad norm-7: {torch.norm(model.features[7].gsp_gate.grad):.3f} | w_norm: {torch.norm(model.features[7].gsp_gate):.3f}")
            print( f"[{batch_idx}/{len(trainloader)}], Loss: {(train_loss/(batch_idx+1))} | Acc: {100.*correct/total}")


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


def zero_filters(model):
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight.data = layer.weight.data * layer.gsp_gate.data

# test(model, criterion, testloader, 1, args)