#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=64g
# --nodelist=trendsdgx001.rs.gsu.edu
#SBATCH --gres=gpu:a100:4
#SBATCH -p qTRDGPUH
#SBATCH -t 7400
#SBATCH -J rohib
#SBATCH -e ./results/zreports/res50-S65-ft80-%A.err
#SBATCH -o ./results/zreports/res50-S65-ft80-%A.out
#SBATCH -A PSYC0002
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rio.ohib@gmail.com

sleep 5s

export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/

source activate imagenet

# ====================================== Baseline Training ======================================
# python main_dali.py -a resnet50 --dist-url 'tcp://127.0.0.1:8800' --dist-backend 'nccl' --epochs 1 \
# --multiprocessing-distributed --world-size 1 --rank 0 /data/users2/rohib/github/imagenet-data


# RESUME from CHECKPOINT
# python main_dali.py -a resnet50 --dist-url 'tcp://127.0.0.1:8800' --dist-backend 'nccl' --epochs 4 \
# --multiprocessing-distributed --world-size 1 --rank 0 /data/users2/rohib/github/imagenet-data \
# --resume checkpoint.pth.tar


# ====================================== GSP from Scratch ======================================
# SPS=0.7
# python main_dali_gsp.py -a resnet50 --dist-url 'tcp://127.0.0.1:8801' --dist-backend 'nccl' \
# --gsp-training --gsp-sps $SPS --resume-lr --batch-size 1024 --epochs 130 --exp-name gsp_S$SPS --lr 0.4  \
# --multiprocessing-distributed --world-size 1 --rank 0 /data/users2/rohib/github/imagenet-data

# GSP Training Continue from CheckPoint
# python main_dali_gsp.py -a resnet50 --dist-url 'tcp://127.0.0.1:8801' --dist-backend 'nccl' \
# --data-backend dali-gpu \
# --gsp-training --gsp-sps $SPS --batch-size 1024 --epochs 130 --exp-name gsp_S$SPS --lr 0.4  \
# --multiprocessing-distributed --world-size 1 --rank 0 /data/users2/rohib/github/imagenet-data \
# --resume ./results/gsp_S$SPS/checkpoint.pth.tar

# ====================================== GSP from FINETUNING ====================================
# # Missing key(s) in state_dict: "module.conv1.mask ...", it means the mask is being added before the
# # sparse trained model is being loaded. Make sure the "if finetuning" block comes after the 
# # "if args.resume block!"
# Also, for Imagenet finetuning, specially with GSP65, starting with LR: 0.00004 works well.
python main_dali_gsp.py -a resnet50 --dist-url 'tcp://127.0.0.1:8801' --dist-backend 'nccl' \
--finetuning --gsp-training --resume-lr --batch-size 1024 --epochs 190 --exp-name gsp_S65_ft85_e20 \
--lr 0.0000004 --finetune-sps 0.85 --multiprocessing-distributed \
--world-size 1 --rank 0 /data/users2/rohib/github/imagenet-data \
--resume ./results/gsp_S65_ft85_e20/checkpoint.pth.tar

# --resume ./results/gsp_S0.65/model_best.pth.tar
# --resume ./results/gsp_S65_ft80_e30/checkpoint.pth.tar
# --resume ./results/gspS65_ft70_lrs/checkpoint.pth.tar
# --resume ./results/gsp_S0.65/model_best.pth.tar


# ================================== Prune and Evaluate Directly ================================
# python main_dali_gsp.py -a resnet50 --dist-url 'tcp://127.0.0.1:8801' --dist-backend 'nccl' \
# --finetuning --gsp-training --resume-lr --batch-size 1024 --epochs 190 --exp-name gsp_S0.65/eval/ --lr 0.04 \
# --multiprocessing-distributed --world-size 1 --rank 0 /data/users2/rohib/github/imagenet-data \
# --evaluate --resume ./results/gsp_S0.65/model_best.pth.tar

# ======================================= EVALUATE CODE =========================================
# python main_dali_gsp.py -a resnet50 --dist-url 'tcp://127.0.0.1:8801' --dist-backend 'nccl' --evaluate \
# --finetuning --gsp-training --resume-lr --batch-size 1024 --epochs 1 --exp-name gsp_S0.65/eval/  --lr 0.4 \
# --multiprocessing-distributed --world-size 1 --rank 0 /data/users2/rohib/github/imagenet-data \
# --evaluate --resume ./results/gsp_S0.65/model_best.pth.tar

# EVALUATE CODE SINGLE GPU
# python main_dali_gsp.py -a resnet50 --dist-url 'tcp://127.0.0.1:8801' --dist-backend 'nccl' --evaluate \
# --gpu 0 --finetuning --gsp-training --resume-lr --batch-size 1024 --epochs 230 --exp-name gsp_S80_fts90/NewEval \
# --lr 0.4 --multiprocessing-distributed --world-size 1 --rank 0 /data/users2/rohib/github/imagenet-data \
# --resume ./results/gsp_S80_fts90/model_best.pth.tar