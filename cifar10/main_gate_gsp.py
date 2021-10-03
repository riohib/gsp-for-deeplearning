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

from models import *

# from utils import progress_bar
from utils.logger import Logger
from gsp_methods import *
import sys 
sys.path.append('/data/users2/rohib/github/testing')
import utils_gsp.sps_tools as sps_tools
from torch.optim.lr_scheduler import MultiStepLR

from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')

parser.add_argument( "--exp-name", metavar="EXPNAME", default="baseline",
                    help="Name of the current experiment",)
parser.add_argument( "--logdir", metavar="LOGDIR", default="/logs",
                    help="directory path for log files",)
parser.add_argument('--gsp-sps', default=0.80, type=float,
                    metavar='SPS', help='gsp sparsity value')            
parser.add_argument('--gsp-int', default=100, type=int,
                    metavar='N', help='GSP projection frequency iteration (default: 100)')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0 

def setup_experiment(args):
    if not os.path.exists(f'./results/{args.exp_name}'):
        os.makedirs(f'./results/{args.exp_name}', exist_ok=True)

    logdir = './results/' + args.exp_name + args.logdir
    print(f'\n input logdir: {logdir} \n')

    args.logger = Logger(logdir, 'logger')
    args.filelogger = args.logger.get_file_logger()
    # args.filelogger.info(f"From: rank: {args.rank} | gpu: {args.gpu}")
    args.writer = SummaryWriter()


def gsp_sparse_training(model, args):
    # Additional Class Variables for GSP
    print(f"ARGS GSP INT: {args.gsp_int}")
    model.sps = args.gsp_sps
    model.curr_iter = 0
    model.start_gsp_epoch = 150
    model.gsp_int = args.gsp_int
    model.logger = args.filelogger
    model.gsp_training_mode = True

    if args.resume:
        model.curr_epoch = args.start_epoch
        print(f"Current Epoch: {args.start_epoch}")
    else:
        model.curr_epoch = 0


def main():
    args = parser.parse_args()
    
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    
    setup_experiment(args)
    
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

    # Model
    print('==> Building model..')
    model = VGG('VGG19')
    # model = Resnet18()

    model = model.to(device)

    if device == 'cuda':
        cudnn.benchmark = True
    #     model = torch.nn.DataParallel(model)
    
    # ===================== BIND GSP METHODS TO MODEL =======================
    gsp_sparse_training(model, args)
    bind_new_gsp_methods_to_model(model, args, apply_gsp=True)


    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/gsp_gates.pth')
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        optimizer.param_groups[0]['lr'] = args.lr

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    
    print(f"Param Opt Len: {len(optimizer.param_groups[0]['params'])}")
    
    # gsp_sparse_training(model, args)
    # bind_gsp_methods_to_model(model, args, apply_gsp=True)

    print(f"ARGS model GSP INT: {model.gsp_int}")

    scheduler = MultiStepLR(optimizer, milestones=[200, 300, 350, 400], gamma=0.1)
    for epoch in range(start_epoch, start_epoch+450):
        train(model, optimizer, criterion, trainloader, epoch, args)
        test(model, criterion, testloader, epoch, args)
        
        model.curr_epoch += 1
        model.curr_iter = 0
        
        scheduler.step()


# Training
def train(model, optimizer, criterion, trainloader, epoch, args):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    model.gsp_training_mode = True

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
            args.filelogger.info(f"Training: epoch:[{epoch}][{batch_idx}/{len(trainloader)}] | Acc@1: {100.*correct/total:.2f} |" \
                f"LR: {optimizer.param_groups[0]['lr']:.5f} | curr_epoch {model.curr_epoch} | curr_itr: {model.curr_iter} " \
                f"grad norm-7: {torch.norm(model.features[7].gsp_gate.grad):.3e} | w_norm: {torch.norm(model.features[7].gsp_gate):.3e}")
        # if batch_idx % 100 == 0:
        #     print(f"Training: epoch:[{epoch}][{batch_idx}/{len(trainloader)}] | Acc@1: {100.*correct/total:.2f} |" \
        #             f"LR: {optimizer.param_groups[0]['lr']:.5f} | curr_epoch {model.curr_epoch} | curr_itr: {model.curr_iter}")

        model.curr_iter += 1
    
    args.writer.add_scalar('val/acc', 100.*correct/total, epoch)

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

    # args.writer.add_scalar('val/acc', acc, epoch)
    args.filelogger.info(f"\nValidation: [{batch_idx}/{len(testloader)}] | Acc@1: {acc:.3f}" \
        f" | Correct: {correct} | total {total} |modelSPS: {sps_tools.get_abs_sps(model)[0]}")

    # print(f"Validation: [{batch_idx}/{len(testloader)}] | Acc@1: {acc:.3f}" \
    #       f"| Correct: {correct} | total {total}")

    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' +str(args.exp_name) + '.pth')
        best_acc = acc

if __name__ == '__main__':
    main()