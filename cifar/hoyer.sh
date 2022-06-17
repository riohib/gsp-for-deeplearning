#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --mem=32g
# --nodelist=trendsdgx001.rs.gsu.edu
#SBATCH --gres=gpu:v100:1
#SBATCH -p qTRDGPUM
#SBATCH -t 1400
#SBATCH -J rohib
#SBATCH -e ./results/zreports/hoyer-bulk-%A.err
#SBATCH -o ./results/zreports/hoyer-bulk-%A.out
#SBATCH -A PSYC0002
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rio.ohib@gmail.com

sleep 5s

export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/

source activate imagenet

model='vgg19_bn'
# ===================================== Train a Baseline Model ======================================
# python main.py --arch vgg16 --batch-size 128 --epochs 200 --lr 0.1 --lr-drop 80 120 160 \
# --exp-name vgg16/baseline

# #===================================== GSP from Scratch ===========================================
# '0.5' '0.6' '0.75' '0.80' '0.85' '0.90' '0.95'
# for reg_decay in '8e-4' '8e-5' '8e-6' '8e-8' '8e-9'
# do
#     python hs_main.py --arch $model --batch-size 128 --epochs 200 --lr 0.1 --lr-drop 80 120 160 \
#     --exp-name hoyersq/reg_${reg_decay} --hoyer-reg
# done

## ==================================== Standalone Finetuning Model ==================================
for sps in  '0.90' '0.95' '0.97' #'0.80' '0.85' '0.90' '0.95' '0.97'
do
    python main.py --arch $model --batch-size 128 --epochs 200 --lr 0.01 --lr-drop 80 140 \
    --exp-name hoyersq/reg_8e-7/sps$sps --finetune --finetune-sps $sps \
    --resume ./results/hoyersq/reg_8e-7/checkpoint.pth.tar
done

## ============================ GO through GSP directories to Finetune ===============================
# for dir in '0.80' '0.85' '0.90' #'0.50' '0.60' '0.70' '0.75'
# do
#     for sps in '0.60' '0.70' '0.80' '0.85' '0.9' '0.95' '0.97' '0.98'
#     do
#         python main.py --arch vgg16 --batch-size 128 --epochs 250 --lr 0.01 --lr-drop 80 120 160 200 \
#         --exp-name vgg16/gspS$dir/fine_$sps --resume ./results/vgg16/gspS$dir/gsp/model_best.pth.tar \
#         --finetune --finetune-sps $sps
#     done
# done

## ====================================================================================================
# rm -rf ./results/gspS$dir/fine_$sps

# for dir in '0.50' '0.60' '0.70' '0.75' '0.80' '0.85' '0.90'
# do
#     for sps in '0.60' '0.70' '0.80' '0.85' '0.9' '0.95' '0.97' '0.98' '0.99'