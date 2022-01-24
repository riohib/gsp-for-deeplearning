# VGG16 Implementation for CIFAR10 in Pytorch

This directory contains the necessary modules to experiment on the Cifar-10 dataset using the VGG16 model.


## How to run?

### To train a baseline model use:
```
python main.py --arch vgg16 --batch-size 128 --epochs 200 --lr 0.1 --lr-drop 80 120 160 --exp-name test/vgg16/baseline
```

<br>

### To train a model with induced GSP every $k$ iterations 
- use `--gsp-int` flag to change the value of $k$.
- all the results will be saved to the directory `--exp-name`
```
python main.py --arch vgg16 --batch-size 128 --epochs 250 --lr 0.1 --lr-drop 80 120 160 200 --exp-name <results dir> --gsp-training --gsp-start-ep 40 --gsp-sps ${SPS}
```

<br>

### To Finetune a induced GSP model with parameters zero or close to zero
```
python main.py --arch vgg16 --batch-size 128 --epochs 250 --lr 0.01 --lr-drop 80 120 160 200 --exp-name $parent/gspS$dir/fine_$sps --finetune --finetune-sps $sps --resume <path to induced GSP model>
```