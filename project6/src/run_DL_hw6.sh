#!/bin/bash
#SBATCH -n 1
#SBATCH -p gputest
#SBATCH -t 0:15:00
#SBATCH --mem=5000
#SBATCH --gres=gpu:v100:1
#SBATCH -J dl6
#SBATCH -o dl6.out.%j
#SBATCH -e dl6.err.%j
#SBATCH --account=project_2002605
#SBATCH   

module purge
module load pytorch/1.1.0

python notebook.py
