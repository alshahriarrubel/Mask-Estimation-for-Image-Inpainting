#!/bin/bash -l
#SBATCH --job-name=maskdetect.predict
#SBATCH --output=%x.%j.out # %x.%j expands to JobName.JobID
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --partition=datasci
#SBATCH --gres=gpu:2
#SBATCH --mem=12G
#SBATCH --mail-user=ar2633@njit.edu
#SBATCH --mail-type=ALL

# Purge any module loaded by default
module purge > /dev/null 2>&1
module load cuda/10.2
module load GCC/9.3.0
conda activate pytorch3d
srun python3 predict.py
module unload cuda/10.2
module unload GCC/9.3.0
