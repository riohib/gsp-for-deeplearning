#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --mem=32g 
# --nodelist=trendsdgx001.rs.gsu.edu
# --gres=gpu:v100:1
#SBATCH --gres=gpu:gforce:1
#SBATCH -p qTRDGPU
#SBATCH -t 1440
#SBATCH -J rohib
#SBATCH -e ./results/zreports/vgg19-rand-%A.err
#SBATCH -o ./results/zreports/vgg19-rand-%A.out
#SBATCH -A PSYC0002
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rio.ohib@gmail.com

sleep 5s

export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/

source activate imagenet


model='vgg19_bn'
parent='vgg/vgg19_bn_ker'

# ===================================== Train a Baseline Model ======================================
# python main.py --arch $model --batch-size 128 --epochs 200 --lr 0.1 --lr-drop 80 120 160 \
# --exp-name $parent/baseline

# ===================================== Random Pruning ======================================
# for sps in '0.80' '0.85' '0.90' '0.95' '0.97'
# do
#     python main.py --arch $model --batch-size 128 --epochs 200 --lr 0.01 --lr-drop 80 140 \
#     --exp-name $parent/random/sps$sps --finetune --finetune-sps $sps --random-pruning \
#     --resume ./results/$parent/baseline/checkpoint.pth.tar
# done

# ===================================== Magnitude Pruning ======================================
for sps in '0.80' '0.85' '0.90' #'0.95' '0.97'
do
    python main.py --arch $model --batch-size 128 --epochs 200 --lr 0.01 --lr-drop 80 140 \
    --exp-name $parent/magnitude/sps$sps --finetune --finetune-sps $sps \
    --resume ./results/$parent/baseline/checkpoint.pth.tar
done

# #===================================== GSP from Scratch ===========================================
# '0.5' '0.6' '0.75' '0.80' '0.85' '0.90' '0.95'
# model='vgg19_bn'
# for SPS in '0.7' '0.80' '0.90'
# do
#     python main.py --arch $model --batch-size 128 --epochs 250 --lr 0.1 --lr-drop 80 120 160 200 \
#     --exp-name vgg/${model}_ker/gspS${SPS}/gsp --gsp-training --gsp-start-ep 40 --gsp-sps ${SPS}
# done

## ==================================== Standalone Finetuning Model ==================================
# for SPS in 0.65 0.75 0.85 0.9 0.95 0.97 0.98 0.99
# do
#     python main.py --arch vgg16 --batch-size 128 --epochs 250 --lr 0.1 --lr-drop 80 120 160 200 \
#     --exp-name gspS80/fine_${SPS} --finetune --finetune-sps ${SPS}
# done

# ## ============================ GO through GSP directories to Finetune ===============================
# # # for dir in '0.50' '0.60' '0.75' '0.80' '0.85' '0.90' '0.95'
# for dir in '0.70'
# do
# #     for sps in '0.60' '0.70' '0.80' '0.85' '0.9' '0.95' '0.97' '0.98'
#     for sps in '0.80' '0.85' '0.90' '0.95' '0.97'
#     do
#         python main.py --arch $model --batch-size 128 --epochs 250 --lr 0.01 --lr-drop 80 120 160 200 \
#         --exp-name $parent/gspS$dir/fine_$sps --finetune --finetune-sps $sps \
#         --resume ./results/$parent/gspS$dir/gsp/model_best.pth.tar \
        
#     done
# done

## ====================================================================================================
# rm -rf ./results/gspS$dir/fine_$sps

# for dir in '0.50' '0.60' '0.70' '0.75' '0.80' '0.85' '0.90'
# do
#     for sps in '0.60' '0.70' '0.80' '0.85' '0.9' '0.95' '0.97' '0.98' '0.99'