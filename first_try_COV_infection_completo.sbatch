#!/bin/bash

#SBATCH --job-name=<COV_fdg>
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:3
#SBATCH --job-name=python-ex
#SBATCH --mail-type=ALL
#SBATCH --partition=cuda

###### 1 Load the module
module load nvidia/cudasdk/11.6
######module load intel/python/3/2019.4.088
# 2 Activate the virtual environment
#source thesis/bin/activate
# CONDA
source ~/.bashrc
conda activate condapip
# 3 Run the python script
python train_COV_fdg.py