#!/bin/bash

#SBATCH --job-name=<CoTrFuse_ISIC2017>
#SBATCH --time=05:00:00
#SBATCH --nodes=2                   # number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
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
python ISIC2017_segmentation_train.py