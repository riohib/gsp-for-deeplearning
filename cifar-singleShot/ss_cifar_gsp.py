'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
from datetime import datetime
import math 

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

import numpy as np

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import util

import logging
import sys
sys.path.append("../")
import utils_gsp.gpu_projection as gsp_gpu
import utils_gsp.padded_gsp as gsp_model
import utils_gsp.sps_tools as sps_tools
import utils_gsp.resnet_tools as res_tools

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('--model', default='model', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--sps', type=float, default=0.99, metavar='SPS',
                    help='gsp sparsity value (default: 0.99)')
parser.add_argument('--targetSps', type=float, default=0.90, metavar='SPS',
                    help='gsp sparsity value (default: 0.99)')

parser.add_argument('--filterwise', type=str, default='Yes', metavar='FW',
                    help='gsp along the filters')

parser.add_argument('--save-dir', type=str, default='./saves/',
                    help='the path to the model saved after training.')
parser.add_argument('--log-dir', type=str, default='.logs/',
                    help='log file name')
parser.add_argument('--prune-method', type=str, default='element', metavar='FW',
                    help='gsp along the filters')
parser.add_argument('--gsp-arch', type=str, default='layerwise', metavar='ARCH',
                    help='gsp along the filters')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# -------------------------- LOGGER ---------------------------------------------------- #
summary_logger = sps_tools.create_logger(args.log_dir, str(args.arch) + str(args.depth)+'_summary')
epoch_logger = sps_tools.create_logger(args.log_dir, str(args.arch) + str(args.depth)+'_training', if_stream = False)
# -------------------------------------------------------------------------------------- #


if args.filterwise == 'Yes':
    is_fw = True
    add_save = '_T'
elif args.filterwise == 'No':
    is_fw = False
    add_save = '_nfw'


summary_logger.info(f"Type of is_FW: {type(args.filterwise)}")

# Generate arg values for summary_logger.infoing with the report:
summary_logger.info("---------------------------------------------------------------------------------")
summary_logger.info("---------------------------------------------------------------------------------")
summary_logger.info(f"All the arguments used are:")
for arg in vars(args):
    summary_logger.info(f"{arg : <20}: {getattr(args, arg)}")
summary_logger.info(f"is_fw : {is_fw:>20}")
summary_logger.info("---------------------------------------------------------------------------------")
summary_logger.info("---------------------------------------------------------------------------------")

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy
save_path = os.path.join(args.save_dir, str(args.arch) + str(args.depth) + add_save)
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)
# else:
#     raise OSError('Directory {%s} exists. Use a new one.' % save_path)

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch


    # Data
    summary_logger.info('==> Preparing dataset %s' % args.dataset)
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
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100


    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader_ = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    trainloader = []
    for i, batch in enumerate(trainloader_):
        inputs, labels = batch[0].cuda(), batch[1].cuda()
        trainloader += [(inputs, labels)]
    
    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader_ = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    testloader = []
    for i, batch in enumerate(testloader_):
        inputs, labels = batch[0].cuda(), batch[1].cuda()
        testloader += [(inputs, labels)]
    
    # Model
    summary_logger.info("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    summary_logger.info('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'cifar-10-' + args.arch
    
    checkpoint_base = torch.load(os.path.join(args.resume, args.model))
    model.load_state_dict( checkpoint_base['state_dict'] )
    
    logger = Logger(os.path.join(save_path, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
    summary_logger.info('model loaded')


    device = torch.device("cuda")
    # Initial training

    # ------------------------ Project the model with GSP: Global vs Layerwise ------------------------ 
    if args.gsp_arch == 'layerwise':
        sps_tools.gsp_resnet_partial(model, args.sps, gsp_func = gsp_gpu, filterwise=is_fw)
    elif args.gsp_arch == 'global':
        xp_mat, ni_list = res_tools.apply_concat_gsp(model, args.sps)
        sps_tools.padded_sparsity(xp_mat, ni_list)
        summary_logger.info(f"The sparsity of the concatenated Matrix is: {sps_tools.padded_sparsity(xp_mat, ni_list)}")
        summary_logger.info(f"The Absolute Sparsity of this model is: {res_tools.get_abs_sps(model)[0].item()}")
        summary_logger.info("------------------- GLOBAL GSP -------------------")
    
    # ------------------------ Structural Pruning for Resnet ------------------------ 
    if args.prune_method == 'structural':
        summary_logger.info("------------------------------- Structural Pruning -------------------------------")
        res_tools.prune_resnet_sps(model, target_sps=args.targetSps)

    summary_logger.info("--- Fixing the Pruned Parameters ---")
    masks = {}
    threshold = 1e-8
    for name, p in model.named_parameters():
        if 'weight' in name:
            tensor = p.data
            masked_tensor = torch.where(abs(tensor) < threshold, torch.tensor(0.0, device=device), tensor)
            mask = torch.where(abs(tensor) < threshold, torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
            masks[name] = mask
            p.data = masked_tensor


    sps_tools.print_nonzeros(model, logger=summary_logger)
    summary_logger.info('Pruned model evaluation...')
    test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
    summary_logger.info(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
    
    if args.evaluate:
        return

    best_prec1 = test_acc
    torch.save(model.state_dict(), os.path.join(save_path, 'finetuned.pth'))

    summary_logger.info("--- Finetuning ---")

    acc_d = {}
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        summary_logger.info('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda, masks)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        if is_best:
            torch.save(model.state_dict(), os.path.join(save_path, 'finetuned.pth'))
            acc_d[epoch] = best_acc

    logger.close()
    #logger.plot()
    #savefig(os.path.join(save_path, 'log.eps'))

    summary_logger.info("--- Evaluating ---")        
    model.load_state_dict(torch.load(os.path.join(save_path, 'finetuned.pth')))
    test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
    summary_logger.info(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
    summary_logger.info('Best acc:')
    summary_logger.info(best_acc)
    summary_logger.info(f"Best Accuracy at epochs: {acc_d}")


def train(trainloader, model, criterion, optimizer, epoch, use_cuda, mask):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        #if use_cuda:
        #    inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        device = torch.device("cuda") 
        for name, p in model.named_parameters():
            if 'weight' in name:
                p.grad.data = p.grad.data*mask[name]
        
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        epoch_logger.info('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    ))
                    
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            #if use_cuda:
            #    inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
        
        nonzero = total = 0
        filter_count = filter_total = 0
        total_sparsity = total_layer = 0
        for name, p in model.named_parameters():
            if 'weight' in name:
                # tensor = p.data.cpu().numpy()
                # tensor = np.abs(tensor)
                # nz_count = np.count_nonzero(tensor)
                # total_params = np.prod(tensor.shape)
                tensor = p.data
                tensor = torch.abs(tensor)
                nz_count = torch.count_nonzero(tensor)
                total_params = tensor.numel()
                
                nonzero += nz_count
                total += total_params
                
        elt_sparsity = (total-nonzero)/total
        summary_logger.info(elt_sparsity)
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
