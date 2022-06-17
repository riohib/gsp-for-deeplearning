#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=32g
#SBATCH -p qTRDGPUL
#SBATCH --nodelist=trendsdgx003.rs.gsu.edu
#SBATCH --gres=gpu:v100:2
#SBATCH -t 500
#SBATCH -J rohib
#SBATCH -e ./reports/baseline/jupyter-%A.err
#SBATCH -o ./reports/baseline/jupyter-%A.out
#SBATCH -A PSYC0002
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rio.ohib@gmail.com

sleep 5s

export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/

source activate tiny-image

# Baseline Training
jupyter lab --no-browser --ip "*" --notebook-dir /data/users2/rohib