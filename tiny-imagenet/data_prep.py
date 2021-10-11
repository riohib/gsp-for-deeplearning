import torch

import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from dali_loader.dataloaders import *
import dali_loader.dali_loaders as Dali

def dataloaders(args):
    #ImageNet: Get PyTorch Dataloaders
    
    dataset = args.dataset
    datadir = args.data

    print(f"dataset: {dataset} | datadir: {datadir}")

    if dataset == 'imagenet_pytorch':
        traindir = os.path.join(datadir, 'train')
        valdir = os.path.join(datadir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize,])

        train_dataset = datasets.ImageFolder(traindir, transform_train)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        transform_val = transforms.Compose([ transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            normalize,
                                            ]) 

        val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(valdir, transform_val),
                                                batch_size=args.batch_size, shuffle=False,
                                                num_workers=args.workers, pin_memory=True)

        return train_loader, len(train_loader), val_loader, len(val_loader)


    # ImageNet: Get DALI Dataloaders
    if dataset == 'imagenet_dali':
        train_loader, train_loader_len, val_loader, val_loader_len = Dali.get_dali_loaders(args)
        return train_loader, train_loader_len, val_loader, val_loader_len
    
    
    # Tiny-ImageNet: Get PyTorch Dataloaders for Tiny-Imagenet dataset
    if dataset == 'tiny_imagenet':
        # Data loading code
        traindir = os.path.join(datadir, 'train')
        valdir = os.path.join(datadir, 'val')
        normalize = transforms.Normalize(mean=[0.480, 0.448, 0.397], std=[0.276, 0.269, 0.282])

        transform_train = transforms.Compose([transforms.RandomCrop(64, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize,])
        train_dataset = datasets.ImageFolder(traindir, transform_train)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        
        transform_val = transforms.Compose([transforms.ToTensor(),normalize,])
        
        val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(valdir, transform_val),
                                                batch_size=args.batch_size, shuffle=False,
                                                num_workers=args.workers, pin_memory=True)
        
        return train_loader, len(train_loader), val_loader, len(val_loader)