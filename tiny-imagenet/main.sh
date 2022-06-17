#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=64g
# --nodelist=trendsdgx001.rs.gsu.edu
#SBATCH --gres=gpu:a100:1
#SBATCH -p qTRDGPUH
#SBATCH -t 7400
#SBATCH -J rohib
#SBATCH -e ./results/base_tiny_res18/res18-baseline-%A.err
#SBATCH -o ./results/base_tiny_res18/res18-baseline-%A.out
#SBATCH -A PSYC0002
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rio.ohib@gmail.com

sleep 5s

export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/

source activate imagenet

# Baseline Training
# python main_dali.py -a resnet50 --dist-url 'tcp://127.0.0.1:8800' --dist-backend 'nccl' --epochs 1 \
# --multiprocessing-distributed --world-size 1 --rank 0 /data/users2/rohib/github/imagenet-data


# RESUME from CHECKPOINT
# python main_dali.py -a resnet50 --dist-url 'tcp://127.0.0.1:8800' --dist-backend 'nccl' --epochs 4 \
# --multiprocessing-distributed --world-size 1 --rank 0 /data/users2/rohib/github/imagenet-data \
# --resume checkpoint.pth.tar

# GSP from Scratch
# python main_dali_gsp.py -a resnet50 --dist-url 'tcp://127.0.0.1:8801' --dist-backend 'nccl' \
# --gsp-training --resume-lr --batch-size 1024 --epochs 130 --exp-name gsp_S80  --lr 0.4  \
# --multiprocessing-distributed --world-size 1 --rank 0 /data/users2/rohib/github/imagenet-data \

# GSP from CHECKPOINT
# python main_dali_gsp.py -a resnet50 --dist-url 'tcp://127.0.0.1:8801' --dist-backend 'nccl' \
# --finetuning --gsp-training --resume-lr --batch-size 1024 --epochs 230 --exp-name gsp_S80_fts85  --lr 0.4 \
# --multiprocessing-distributed --world-size 1 --rank 0 /data/users2/rohib/github/imagenet-data \
# --resume ./results/gsp_S80/model_best.pth.tar

# Tiny Imagenet
python main_dali_gsp.py -a resnet18 --batch-size 256 --epochs 200 --exp-name base_tiny_res18 --lr 0.2 \
--lr-drop 100 150 --dataset tiny_imagenet /data/users2/rohib/datasets/tiny-imagenet-200
# /data/users2/rohib/github/imagenet-data
# /data/users2/rohib/datasets/tiny-imagenet-200

# python main_dali_gsp.py -a resnet50 --dist-url 'tcp://127.0.0.1:8801' --dist-backend 'nccl' \
# --gsp-training --resume-lr --batch-size 1024 --epochs 130 --exp-name gsp_S80_DP  --lr 0.4  \
# --world-size 1 --rank 0 /data/users2/rohib/github/imagenet-data 


# EVALUATE CODE
# python main_dali_gsp.py -a resnet50 --dist-url 'tcp://127.0.0.1:8803' --dist-backend 'nccl' \
# --finetuning --gsp-training --resume-lr --batch-size 1024 --epochs 230 --exp-name gsp_S80_ft  --lr 0.4 \
# --multiprocessing-distributed --world-size 1 --rank 0 /data/users2/rohib/github/imagenet-data \
# --resume post_gsp.pth.tar
