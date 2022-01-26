# ImageNet training in PyTorch

### Train a baseline ResNet50 Model on ImageNet
```
python main_dali.py -a resnet50 --dist-url 'tcp://127.0.0.1:8800' --dist-backend 'nccl' --epochs 90 --multiprocessing-distributed --world-size 1 --rank 0 <path to ImageNet data>
```

<br>

### Train a ResNet50 model with Induced GSP
SPS=0.7
```
python main_dali_gsp.py -a resnet50 --dist-url 'tcp://127.0.0.1:8801' --dist-backend 'nccl' --gsp-training --gsp-sps $SPS --resume-lr --batch-size 1024 --epochs 130 --exp-name gsp_S$SPS --lr 0.4 --multiprocessing-distributed --world-size 1 --rank 0 <path to ImageNet data>
```

<br>

### Finetune the trained Induced GSP model
The epoch counter when finetuning a trained induced gsp model might start from the last epoch of induced gsp training. Please add the required finetuning epoch to the previously trained epoch to the `--epochs` flag.

```
python main_dali_gsp.py -a resnet50 --dist-url 'tcp://127.0.0.1:8801' --dist-backend 'nccl' --finetuning --gsp-training --resume-lr --batch-size 1024 --exp-name <results dir> --multiprocessing-distributed --world-size 1 --rank 0 <path to ImageNet data > --resume <Path to Induced GSP model>
```

<br>

### To evaluate a trained model
```
python main_dali_gsp.py -a resnet50 --dist-url 'tcp://127.0.0.1:8801' --dist-backend 'nccl' --evaluate --gpu 0 --finetuning --gsp-training --resume-lr --batch-size 1024 --epochs 230 --exp-name gsp_S80_fts90/eval --lr 0.4 --multiprocessing-distributed --world-size 1 --rank 0 <path to ImageNet data> --resume <path to model>
```