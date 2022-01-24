## Single Shot GSP and Finetuning on CIFAR-10

We use the `ss_cifar_gsp.py` module to project the model weights of a model once (single shot) and then finetune the remaining weights. We use:

```
python ss_cifar_gsp.py -a resnet --depth $M_DEPTH --epochs 200 --schedule 81 122 174 --gamma 0.1 \
--wd 1e-4 --resume $MODEL_PATH --model model_best.pth.tar --sps $SPS --targetSps $SPS 
--filterwise 'Yes' --save-dir $SAVEDIR --log-dir $LOGDIR
```
where `$MODEL_PATH` should point to the directory of a already trained model. 