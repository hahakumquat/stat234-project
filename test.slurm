#!/bin/bash
#
#SBATCH -N 1
#SBATCH -t 5-00:00
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michaelge@college.harvard.edu, rouyang@college.harvard.edu

python test.py