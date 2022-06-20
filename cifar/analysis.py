import torch
import torch.nn as nn
from main import AverageMeter, accuracy

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