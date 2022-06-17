#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --mem=32g
# --nodelist=trendsdgx001.rs.gsu.edu
#SBATCH --gres=gpu:v100:1
#SBATCH -p qTRDGPUM
#SBATCH -t 7400
#SBATCH -J rohib
#SBATCH -e ./results/zreports/res32-gsp-%A.err
#SBATCH -o ./results/zreports/res32-gsp-%A.out
#SBATCH -A PSYC0002
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rio.ohib@gmail.com

sleep 5s

export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/

source activate imagenet


# ===================================== Train a Baseline Model ======================================
# python main.py --arch resnet56 --batch-size 128 --epochs 200 --lr 0.1 --lr-drop 80 120 160 \
# --exp-name resnets/resnet56/baseline


# ===================================== Random Pruning ======================================
# python main.py --arch resnet56 --batch-size 128 --epochs 200 --lr 0.1 --lr-drop 80 120 160 \
# --exp-name resnet56/baseline
# for sps in '0.8'
# do
#     python main.py --arch resnet56 --batch-size 128 --epochs 200 --lr 0.01 --lr-drop 80 120 160 \
#     --exp-name random/sps$sps --finetune --finetune-sps $sps --random-pruning \
#     --resume ./results/$parent/gspS$dir/gsp/model_best.pth.tar
# done

# ===================================== Magnitude Pruning ======================================
# python main.py --arch resnet56 --batch-size 128 --epochs 200 --lr 0.1 --lr-drop 80 120 160 \
# --exp-name resnet56/baseline

# for sps in '0.80' '0.85' '0.90' '0.95' '0.97'
# do
#     python main.py --arch resnet56 --batch-size 128 --epochs 200 --lr 0.01 --lr-drop 80 120 160 \
#     --exp-name magnitude/sps$sps --finetune --finetune-sps $sps \
#     --resume ./results/resnet56_filter/baseline/checkpoint.pth.tar
# done

# #===================================== GSP from Scratch ===========================================
# '0.5' '0.6' '0.75' '0.80' '0.85' '0.90' '0.95'
# model='resnet32'
# for SPS in '0.80'
# do
#     python main.py --arch $model --batch-size 128 --epochs 250 --lr 0.1 --lr-drop 80 120 160 200 \
#     --exp-name resnets/${model}_ker/gspS${SPS}/gsp --gsp-training --gsp-start-ep 40 --gsp-sps ${SPS}
# done

## ==================================== Standalone Finetuning Model ==================================
# for SPS in 0.65 0.75 0.85 0.9 0.95 0.97 0.98 0.99
# do
#     python main.py --arch resnet56 --batch-size 128 --epochs 250 --lr 0.1 --lr-drop 80 120 160 200 \
#     --exp-name gspS80/fine_${SPS} --finetune --finetune-sps ${SPS}
# done

# ## ============================ GO through GSP directories to Finetune ===============================
model='resnet32'
parent='resnets/resnet32_ker'
for dir in '0.80'
do
    for sps in '0.80' '0.85' '0.90' '0.95' '0.97'
    do
        python main.py --arch $model --batch-size 128 --epochs 250 --lr 0.001 --lr-drop 80 120 160 200 \
        --exp-name $parent/gspS$dir/fine_$sps --finetune --finetune-sps $sps \
        --resume ./results/$parent/gspS$dir/gsp/model_best.pth.tar    
    done
done