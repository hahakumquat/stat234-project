#!/bin/bash
#
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 5-00:00
#SBATCH -p gpu_requeue
#SBATCH --mem=1024
#SBATCH -o gpu.out
#SBATCH -e gpu.err
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --mail-user=michaelge@college.harvard.edu, rouyang@college.harvard.edu

python test.py