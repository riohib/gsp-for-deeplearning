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
import networks.resnet as resnet
from torch.optim.lr_scheduler import MultiStepLR

from datetime import datetime

import sys 
sys.path.append('..')
from utils_gsp.logger import Logger
# from utils_gsp import sps_tools
from gsp_model import GSP_Model, hoyer_square, group_hoyer_sq

import networks.load as load

from torch.utils.tensorboard import SummaryWriter



parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    help='model architecture to use!')
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
parser.add_argument("--lr-drop", nargs="*", type=int, default=[100, 150],)
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
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)

parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument( "--exp-name", metavar="EXPNAME", default="baseline",
                    help="Name of the current experiment",)
parser.add_argument( "--logdir", metavar="LOGDIR", default="/logs",
                    help="directory path for log files",)

parser.add_argument('--baseline', action='store_true',
                    help='Train a baseline dense model!')

parser.add_argument('--gsp-training', action='store_true',
                    help='Train the model with gsp projection every few iteration (--gsp-int)')
parser.add_argument('--gsp-sps', default=0.8, type=float,
                    metavar='SPS', help='gsp sparsity value')
parser.add_argument('--proj-model',  action='store_true',
                    help='Projects all the layers of model simultaneously')
parser.add_argument('--proj-filters',  action='store_true',
                    help='Projects the cnn filters when flagged, otherwise projects along kernel positions')
parser.add_argument('--gsp-int', default=150, type=int,
                    metavar='GSPINT', help='GSP projection frequency iteration (default: 500)')
parser.add_argument('--gsp-start-ep', default=-1, type=int,
                    metavar='GSPSTART', help='Epoch to start gsp projection')

parser.add_argument('--finetune', action='store_true',
                    help='Finetune a select subset of parameters of a sparse model.')
parser.add_argument('--finetune-sps', default=0.85, type=float,
                    metavar='SPS', help='gsp sparsity value')

parser.add_argument('--hoyer-reg', action='store_true',
                    help='Train the model with Hoyer Square Regularizer')
parser.add_argument('--reg-decay', default=8e-4, type=float,
                    metavar='REGDECAY', help='Hoyer Square Regularizer Weight Parameter')


best_acc1 = 0
best_epoch = 0


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
    model.logger = args.filelogger

    # Extract modes for GSP
    model.start_gsp_epoch = args.gsp_start_ep
    model.gsp_int = args.gsp_int
    model.sps = args.gsp_sps
    model.project_model = args.proj_model
    model.proj_filters = args.proj_filters

    args.filelogger.info(f"GSP_model State: sps: {model.sps} | start_ep: {model.start_gsp_epoch} | " \
                         f"Interval: {model.gsp_int} | Model Proj: {model.project_model} | " \
                         f"Filter Proj: {model.proj_filters}")

    if args.resume:
        model.curr_epoch = args.start_epoch
        print(f"Current Epoch: {args.start_epoch}")
    else:
        model.curr_epoch = 0


def main():
    global args, best_acc1
    args = parser.parse_args()
    
    # torch.manual_seed(0)
    # Setup the experiment
    flogger = setup_experiment(args)
    args.logger.log_cmd_arguments(args)

    # Load Model
    model = load.model(args.arch)
    model = torch.nn.DataParallel(model)
    model.cuda()

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

    flogger.info(f"The sparsity of Initialized Model: {model_gsp.get_model_sps():.2f}")
    args.writer = SummaryWriter(log_dir=f'results/{args.exp_name}/runs/{datetime.now().strftime("%m-%d_%H:%M")}')
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            flogger.info("=> Loading Model from CheckPoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] if not args.finetune else 0
            best_acc1 = checkpoint['best_acc1'] if not args.finetune else 0
            model.load_state_dict(checkpoint['state_dict'])
            flogger.info(f"=> loaded checkpoint at succesfully from (epoch {checkpoint['epoch']})")
            validate(val_loader, model_gsp.model, criterion, args)
            flogger.info(f"The sparsity of the loaded model is: {model_gsp.get_model_sps():.2f}")
        else:
            print("*=> LOADING FAILED: no checkpoint found at '{}'".format(args.resume))


    # ============================ Setup GSP model ============================        
    if args.gsp_training and not args.baseline:
        gsp_sparse_training(model_gsp, train_loader, args)
        flogger.info(15*"*" + " Model will be trained with GSP Sparsity!! " + 15*"*" )

    # ============== PRUNE the model and Register Mask ==============
    if args.finetune and not args.baseline:
        flogger.info(15*"*" + " Model will be finetuned!! " + 15*"*")
        model_gsp.prune_and_mask_model(sps=args.finetune_sps)
    
    if args.gsp_training and not args.baseline:
        flogger.info(15*"*" + " Model will be regularized with HoyerSquare " + 15*"*" )


    if args.evaluate:
        validate(val_loader, model, criterion)
        return



    scheduler = MultiStepLR(optimizer, milestones=args.lr_drop, last_epoch=args.start_epoch - 1)
    for epoch in range(args.start_epoch, args.epochs):
    
        train(train_loader, model_gsp, criterion, optimizer, epoch, args) # Train model
        prec1 = validate(val_loader, model_gsp.model, criterion, args) # evaluate on validation set

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_acc1
        best_acc1 = max(prec1, best_acc1)
        if is_best:
            best_epoch = epoch 

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
    
    # Exit Metrics
    if args.finetune:
        pre1 = validate(val_loader, model_gsp.model, criterion, args)
        flogger.info(f"Validation accuracy post training: {pre1}")
        model_gsp.mask_out_parameters() # Force Masked Parameters to zero
        post1 = validate(val_loader, model_gsp.model, criterion, args)
        flogger.info(f"Validation accuracy post force masking parameters: {post1}")

    # Exit
    flogger.info(f"\n Final Model SPS: {model_gsp.get_model_sps():.2f}% | Best @acc: {best_acc1} achieved in epoch: {best_epoch}")


def train(train_loader, model_gsp, criterion, optimizer, epoch, args, gsp_mode=None):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model_gsp.model.train()
    model_gsp.gsp_training_mode = True if (gsp_mode==None) else gsp_mode
    flogger = args.filelogger
    len_trainloader = len(train_loader)
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
        if args.gsp_training and not args.finetune and not args.hoyer_reg:
            model_gsp.apply_gsp()
        # ===================================================================

        # compute output
        output = model_gsp.model(input_var)
        loss = criterion(output, target_var)

        # --------------------------- Hoyer Square ---------------------------
        if args.hoyer_reg and not args.gsp_training and not args.finetune:
            orig_loss, loss = hoyer_square(model_gsp.model, args.reg_decay, loss)

        if i % len_trainloader == 0:
            flogger.info(f"modelSPS before optimization: {model_gsp.get_model_sps():.2f}%")

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
            flogger.info(f"Training: epoch:[{epoch}][{i}/{len(train_loader)}] | Acc@1: {top1.avg:.2f}" \
                f" | OrigLoss: {orig_loss:.3f} | loss: {loss:.3f}" \
                f" | LR: {optimizer.param_groups[0]['lr']:.5f} | Mcurr_epoch {model_gsp.curr_epoch}/{epoch}" \
                f" | Mcurr_itr: {model_gsp.curr_iter-1}/{i} | modelSPS: {model_gsp.get_model_sps():.2f}%")
        

def validate(val_loader, model, criterion, args):
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


    args.filelogger.info(f"\n Validation Acc@1: {top1.avg:.3f} \n")

    return top1.avg


def dataloader_to_gpu(train_loader, val_loader):
    train_loader_cuda = list()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        train_loader_cuda.append((inputs, labels))

    val_loader_cuda = list()
    for i, (inputs, labels) in enumerate(val_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        val_loader_cuda.append((inputs, labels))

    return train_loader_cuda, val_loader_cuda


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