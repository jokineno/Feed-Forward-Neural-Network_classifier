#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 0:30:00
#SBATCH --mem=5000
#SBATCH --gres=gpu:v100:1
#SBATCH -J dl5
#SBATCH -o dl5.out.%j
#SBATCH -e dl5.err.%j
#SBATCH --account=project_2002605
#SBATCH   

module purge
module load pytorch/1.1.0

python DL_hw5.py
