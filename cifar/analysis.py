import torch
import torch.nn as nn

class ModelAnalysis:
    def __init__(self, model, optimizer, criterion, train_loader, test_loader):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader


    def setup_meters(self):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
    
    def re_init_model(self):
        print('=> weights being re-initialized')
        self.model.apply(weights_init)

    def train_mode(self):
        if not self.model.training:
            print(f"Setting to train mode")
            self.model.train()
    
    def validate(self):
        """Run evaluation"""
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.cuda(), target.cuda()

                # compute output
                output = self.model(data)
                loss = self.criterion(output, target)

                output = output.float()
                loss = loss.float()

                # measure accuracy and record loss
                prec1 = accuracy(output.data, target)[0]
                losses.update(loss.item(), data.size(0))
                top1.update(prec1.item(), data.size(0))
        print(f"\n Validation Acc@1: {top1.avg:.3f} \n")
        return top1.avg
    
    def sample_data(self):
        data, target = next(iter(self.train_loader))
        data, target = data.cuda(), target.cuda()
        return data, target
    
    def sample_forward(self):
        data, target = self.sample_data()
        output = self.model(data)
        loss = self.criterion(output, target)
        return loss
    
    def forward(self, data, target):
        output = self.model(data)
        loss = self.criterion(output, target)
        return loss

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
    
    def step(self):
        self.optimizer.step()

    def iterate_once(self):
        self.train_mode()
        loss = self.sample_forward()
        self.backward(loss)
        self.step()
        return loss

    def train_epoch(self):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()  
        self.model.train()
        
        for i, (data, target) in enumerate(self.train_loader):
            data, target = data.cuda(), target.cuda()
            output = self.model(data)
            loss = self.criterion(output, target)
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            output = output.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            if i % 20== 0:
                print(f"Training: batch_id:[{i}] | Acc@1: {top1.avg:.2f} | Loss: {loss.item()}")
    

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        # Note that BN's running_var/mean are already initialized to 1 and 0 respectively.
        if m.weight is not None:
            m.weight.data.fill_(1.0)
        if m.bias is not None:
            m.bias.data.zero_()

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