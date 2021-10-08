import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet

from datetime import datetime

import sys 
sys.path.append('..')
from utils_gsp.logger import Logger
from utils_gsp import sps_tools
from gsp_model import GSP_Model

from torch.utils.tensorboard import SummaryWriter

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=160, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)

parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument( "--exp-name", metavar="EXPNAME", default="baseline",
                    help="Name of the current experiment",)
parser.add_argument( "--logdir", metavar="LOGDIR", default="/logs",
                    help="directory path for log files",)

parser.add_argument('--gsp-training', action='store_true',
                    help='Train the model with gsp projection every few iteration (--gsp-int)')
parser.add_argument('--gsp-sps', default=0.8, type=float,
                    metavar='SPS', help='gsp sparsity value')
parser.add_argument('--gsp-int', default=150, type=int,
                    metavar='N', help='GSP projection frequency iteration (default: 500)')
parser.add_argument('--gsp-start-ep', default=-1, type=int,
                    metavar='N', help='Epoch to start gsp projection')

parser.add_argument('--finetuning', action='store_true',
                    help='Finetune a select subset of parameters of a sparse model.')
parser.add_argument('--finetune-sps', default=0.85, type=float,
                    metavar='SPS', help='gsp sparsity value')

best_acc1 = 0


def main():
    global args, best_acc1
    args = parser.parse_args()
    
    # torch.manual_seed(0)
    # Setup the experiment
    flogger = setup_experiment(args)
    args.logger.log_cmd_arguments(args)

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    # ----------------------- Make a GSP Model -----------------------
    model_gsp = GSP_Model(model)

    flogger.info(f"The sparsity of the model is: {model_gsp.get_model_sps():.2f}")
    args.writer = SummaryWriter(log_dir=f'results/{args.exp_name}/runs/{datetime.now().strftime("%m-%d_%H:%M")}')
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # ============================ Setup GSP model ============================
    if args.gsp_training:
        gsp_sparse_training(model_gsp, train_loader, args)
        flogger.info(15*"*" + " Model will be trained with GSP Sparsity!! " + 15*"*" )

    # ============== PRUNE the model and Register Mask ==============
    if args.finetuning:
        flogger.info(15*"*" + " Model will be finetuned!! " + 15*"*")
        sps_tools.prune_with_sps(model_gsp.model.module, sparsity = args.finetune_sps)
        masks_d, masks_l = sps_tools.get_conv_linear_mask(model_gsp.model.module)
        model_gsp.register_pre_hook_mask(masks_d) # This for forward pre hook mask registration
        # model_gsp.register_hook_mask(model.module, masks_l) # Does not work with DDP


    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                    milestones=[80, 120], last_epoch=args.start_epoch - 1)

    
    for epoch in range(args.start_epoch, args.epochs):
    
        train(train_loader, model_gsp, criterion, optimizer, epoch) # Train model
        prec1 = validate(val_loader, model, criterion) # evaluate on validation set

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_acc1
        best_acc1 = max(prec1, best_acc1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model_gsp.model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args, filename=  f'results/{args.exp_name}/checkpoint.pth.tar')

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model_gsp.model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args, filename=  f'results/{args.exp_name}/checkpoint.pth.tar')
        
        scheduler.step()


def train(train_loader, model_gsp, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model_gsp.model.train()
    model_gsp.gsp_training_mode = True
    flogger = args.filelogger

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # ============================ Apply GSP ============================
        if args.gsp_training:
            model_gsp.apply_gsp()
        # ===================================================================

        # compute output
        output = model_gsp.model(input_var)
        loss = criterion(output, target_var)

        if i % 390 == 0:
            flogger.info(f"------ modelSPS before optimization: {model_gsp.get_model_sps():.2f}%")

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            flogger.info(f"Training: epoch:[{epoch}][{i}/{len(train_loader)}] | Acc@1: {top1.avg:.2f} |" \
                f"LR: {optimizer.param_groups[0]['lr']:.5f} | Mcurr_epoch {model_gsp.curr_epoch}/{epoch}"\
                f" | Mcurr_itr: {model_gsp.curr_iter}/{i} | modelSPS: {model_gsp.get_model_sps():.2f}%")


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model.gsp_training_mode = False

    end = time.time()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


    args.filelogger.info(f"\n Validation: [{batch_idx}/{len(val_loader)}] | Acc@1: {top1.avg:.3f} \n")

    return top1.avg


def setup_experiment(args):
    if not os.path.exists(f'./results/{args.exp_name}'):
        os.makedirs(f'./results/{args.exp_name}', exist_ok=True)

    logdir = './results/' + args.exp_name + args.logdir
    print(f'\n input logdir: {logdir} \n')

    args.logger = Logger(logdir, f'gpu:{args.gpu}')
    args.filelogger = args.logger.get_file_logger()
    # args.filelogger.info(f"From: rank: {args.rank} | gpu: {args.gpu}")
    return args.filelogger

def gsp_sparse_training(model, train_loader, args):
    # Additional Class Variables for GSP
    model.curr_iter = 0
    model.train_loader_len = len(train_loader)

    model.sps = args.gsp_sps
    model.start_gsp_epoch = args.gsp_start_ep
    model.gsp_int = args.gsp_int
    model.logger = args.filelogger

    args.filelogger.info(f"GSP_model State: sps: {model.sps} | start_ep: {model.start_gsp_epoch} | Interval: {model.gsp_int}")

    if args.resume:
        model.curr_epoch = args.start_epoch
        print(f"Current Epoch: {args.start_epoch}")
    else:
        model.curr_epoch = 0


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'./results/{args.exp_name}/model_best.pth.tar')
        

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()