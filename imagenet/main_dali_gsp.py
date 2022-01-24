import argparse
from genericpath import exists
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
from torch.optim.lr_scheduler import MultiStepLR

from datetime import datetime
import time

from torch.utils.tensorboard import SummaryWriter
from dali_loader.dataloaders import *
import dali_loader.dali_loaders as Dali

# from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
from utils.logger import Logger
import models.resnet_torch as ResNet

import sys 
sys.path.append('/data/users2/rohib/github/testing')
import utils_gsp.sps_tools as sps_tools
# import utils_gsp.gpu_projection as gsp_gpu
from apply_gsp import GSP_Model

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

parser.add_argument( "--data-backend", metavar="BACKEND", default="dali-cpu", 
                    choices=DATA_BACKEND_CHOICES,
                    help="data backend: "+ " | ".join(DATA_BACKEND_CHOICES) + " (default: dali-cpu)",)
parser.add_argument( "--interpolation", metavar="INTERPOLATION", default="bilinear",
                    help="interpolation type for resizing images: bilinear, bicubic or triangular(DALI only)",)
parser.add_argument('-is', '--image-size', default=224, type=int, metavar='N',
                    help='default image size (default: 224)')
parser.add_argument("--mixup", default=0.0, type=float, metavar="ALPHA", help="mixup alpha")
parser.add_argument('--num-classes', default=1000, type=int, metavar='N',
                    help='Imagenet number of classes (default: 1000)')
parser.add_argument("--augmentation", type=str, default=None, 
                    choices=[None, "autoaugment"], help="augmentation method",)
parser.add_argument("--memory-format", type=str, default="nchw",
                    choices=["nchw", "nhwc"], help="memory layout, nchw or nhwc",)


parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--resume-lr', dest='resume_lr', action='store_true',
                    help='Forces the resume checkpoing to start with new learning rate.')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=250, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument( "--exp-name", metavar="EXPNAME", default="baseline",
                    help="Name of the current experiment",)

parser.add_argument( "--logdir", metavar="LOGDIR", default="/logs",
                    help="directory path for log files",)

parser.add_argument('--gsp-training', action='store_true',
                    help='Train the model with gsp projection every few iteration (--gsp-int)')
parser.add_argument('--gsp-sps', default=0.8, type=float,
                    metavar='SPS', help='gsp sparsity value')
parser.add_argument('--gsp-int', default=500, type=int,
                    metavar='N', help='GSP projection frequency iteration (default: 500)')

parser.add_argument('--finetuning', action='store_true',
                    help='Finetune a select subset of parameters of a sparse model.')
parser.add_argument('--finetune-sps', default=0.70, type=float,
                    metavar='SPS', help='gsp sparsity value')
best_acc1 = 0


def main():
    args = parser.parse_args()   

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
        main_worker(args.gpu, ngpus_per_node, args)


def setup_experiment(args):
    if not os.path.exists(f'./results/{args.exp_name}'):
        os.makedirs(f'./results/{args.exp_name}', exist_ok=True)

    logdir = './results/' + args.exp_name + args.logdir
    print(f'\n input logdir: {logdir} \n')

    args.logger = Logger(logdir, f'gpu:{args.gpu}')
    args.filelogger = args.logger.get_file_logger()
    args.filelogger.info(f"From: rank: {args.rank} | gpu: {args.gpu}")
    return args.filelogger

def gsp_sparse_training(model, args):
    # Additional Class Variables for GSP
    model.sps = args.gsp_sps
    model.curr_iter = 0
    model.start_gsp_epoch = 40
    model.gsp_int = args.gsp_int
    model.logger = args.filelogger

    if args.resume:
        model.curr_epoch = args.start_epoch
        print(f"Current Epoch: {args.start_epoch}")
    else:
        model.curr_epoch = 0


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    flogger = setup_experiment(args)

    args.logger.log_cmd_arguments(args)

    if args.gpu is not None:
        flogger.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    if args.pretrained:
        flogger.info("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        flogger.info("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=False)
        # model = ResNet.resnet50(pretrained=False)
        # model = ResNet.__dict__[args.arch](pretrained=False)
        flogger.info("Created model from Pytorch Model Zoo!")


    if not torch.cuda.is_available():
        flogger.info('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        print("In the DataParallel only zone!")
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # ----------------------- optionally resume from a checkpoint -----------------------
    model_gsp = GSP_Model(model) # Make a GSP Model

    flogger.info(f"The sparsity of the model is: {sps_tools.get_abs_sps(model_gsp.model)[0]:.2f}")
    cudnn.benchmark = True

    # Get DALI Dataloaders
    train_loader, train_loader_len, val_loader, val_loader_len = Dali.get_dali_loaders(args)


    # Create Summary Writer
    if args.gpu == 0:
        args.writer = SummaryWriter(log_dir=f'results/{args.exp_name}/runs/{datetime.now().strftime("%m-%d_%H:%M")}')

    # Setup GSP model
    gsp_sparse_training(model_gsp, args)
    if args.gsp_training:
        flogger.info(15*"*" + " Model will be trained with GSP Sparsity!! " + 15*"*" )

    if args.resume:
        if os.path.isfile(args.resume):
            flogger.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model_gsp.model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            flogger.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            flogger.info(f"Loaded State Dict: LR: {optimizer.param_groups[0]['lr']:.5f} |" \
                                 f"Best @acc1: {checkpoint['best_acc1']}")
            flogger.info(f"The sparsity of the loaded model: {sps_tools.get_abs_sps(model)[0]:.2f}")
        else:
            flogger.info("=> no checkpoint found at '{}'".format(args.resume))


    # PRUNE the model and Register Mask
    if args.finetuning:
        flogger.info(15*"*" + f" Model will be finetuned with sps: {args.finetune_sps} " + 15*"*")
        sps_tools.prune_with_sps(model_gsp.model.module, sparsity = args.finetune_sps)
        masks_d, masks_l = sps_tools.get_conv_linear_mask(model_gsp.model.module)
        model_gsp.register_pre_hook_mask(masks_d) # This for forward pre hook mask registration
        # model_gsp.register_hook_mask(model.module, masks_l) # Does not work with DDP


    if args.resume_lr:
        flogger.info(f"Resuming with new Learning Rate! {args.lr}")
        optimizer.param_groups[0]['lr'] = args.lr
        
    if args.evaluate:
        flogger.info(f"The sparsity of the model to be validated: {sps_tools.get_abs_sps(model_gsp.model)[0]:.2f}")
        validate(val_loader, val_loader_len, model_gsp.model, criterion, args)
        return

    scheduler = MultiStepLR(optimizer, milestones=[70, 100], gamma=0.1)
    for epoch in range(args.start_epoch, args.epochs):

        t0 = time.time() #timing

        # TRAIN for one epoch
        acc1, loss = train(train_loader, train_loader_len, model_gsp, criterion, optimizer, epoch, args)
        t1 = time.time() - t0
        flogger.info(f"Epoch {epoch} took {t1} seconds")
        
        if args.gpu == 0:
            args.writer.add_scalar('train/loss', loss.item(), epoch)
            args.writer.add_scalar('train/acc', acc1, epoch)

        # evaluate on VALIDATION set
        acc1, loss = validate(val_loader, val_loader_len, model_gsp.model, criterion, args)

        if args.gpu == 0:
            args.writer.add_scalar('val/loss', loss.item(), epoch)
            args.writer.add_scalar('val/acc', acc1, epoch)
 
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # save_dir
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model_gsp.model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args, filename=  f'results/{args.exp_name}/checkpoint.pth.tar')

        model_gsp.curr_epoch += 1
        model_gsp.curr_iter = 0
        scheduler.step()

    flogger.info(f"The sparsity of the model post training is: {sps_tools.get_abs_sps(model)[0]:.2f}")


def train(train_loader, train_loader_len, gsp_model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        train_loader_len,
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    
    
    gsp_model.gsp_training_mode = False if args.finetuning else True

    # switch to train mode
    gsp_model.model.train()
    flogger = args.filelogger

    end = time.time()
    for batch_idx, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # ============================ Apply GSP ============================
        if args.gsp_training:
            gsp_model.apply_gsp()
        # ===================================================================

        # compute output
        output = gsp_model.model(images)
        loss = criterion(output, target)
    
        if batch_idx % args.print_freq == 0:
            flogger.info(f"------ modelSPS before optimization: {gsp_model.get_model_sps():.2f}%")

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            progress.display(batch_idx)
            flogger.info(f"Training: epoch:[{epoch}][{batch_idx}/{train_loader_len}] | Acc@1: {acc1.item():.2f} |" \
                f"Acc@5: {acc5.item():.2f} | LR: {optimizer.param_groups[0]['lr']:.5f} | curr_epoch {gsp_model.curr_epoch}"\
                f" | curr_itr: {gsp_model.curr_iter} | modelSPS: {gsp_model.get_model_sps():.2f}%")
        
        gsp_model.curr_iter += 1

    return acc1, loss

def validate(val_loader, val_loader_len, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        val_loader_len,
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    # Turn off GSP
    model.gsp_training_mode = False 

    with torch.no_grad():
        end = time.time()
        for batch_idx, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if batch_idx % args.print_freq == 0:
            #     progress.display(batch_idx)
                # args.filelogger.info(f"Validation: [{batch_idx}/{val_loader_len}] | Acc@1: {top1.avg:.3f} | Acc@5: {top5.avg:.3f}")

    
    args.filelogger.info(f"\n The mean Model Accuracy Acc@1: {top1.avg:.3f} | Acc@5: {top5.avg:.3f}")
    args.filelogger.info(80*"-")

    return top1.avg, loss



def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'./results/{args.exp_name}/model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# def adjust_learning_rate(optimizer, epoch, args):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    main()

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)