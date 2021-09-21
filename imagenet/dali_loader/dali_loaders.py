
import os
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from functools import partial

from dali_loader.dataloaders import *
# from dali_loader.autoaugment import AutoaugmentImageNetPolicy


def get_dali_loaders(args):
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    start_epoch = 0

    memory_format = (
        torch.channels_last if args.memory_format == "nhwc" else torch.contiguous_format
    )
    image_size = (
        args.image_size
        if args.image_size is not None
        else model.arch.default_image_size
    )
    # Create data loaders and optimizers as needed
    if args.data_backend == "pytorch":
        get_train_loader = get_pytorch_train_loader
        get_val_loader = get_pytorch_val_loader
    elif args.data_backend == "dali-gpu":
        get_train_loader = get_dali_train_loader(dali_cpu=False)
        get_val_loader = get_dali_val_loader()
    elif args.data_backend == "dali-cpu":
        get_train_loader = get_dali_train_loader(dali_cpu=True)
        get_val_loader = get_dali_val_loader()
    elif args.data_backend == "syntetic":
        get_val_loader = get_syntetic_loader
        get_train_loader = get_syntetic_loader
    else:
        print("Bad databackend picked")
        exit(1)
    
    train_loader, train_loader_len = get_train_loader(
        args.data,
        image_size,
        args.batch_size,
        args.num_classes,
        args.mixup > 0.0,
        interpolation = args.interpolation,
        augmentation=args.augmentation,
        start_epoch=start_epoch,
        workers=args.workers,
        memory_format=memory_format,
    )

    if args.mixup != 0.0:
        train_loader = MixUpWrapper(args.mixup, train_loader)

    val_loader, val_loader_len = get_val_loader(
        args.data,
        image_size,
        args.batch_size,
        args.num_classes,
        False,
        interpolation = args.interpolation,
        workers=args.workers,
        memory_format=memory_format,
    )

    return train_loader, train_loader_len, val_loader, val_loader_len