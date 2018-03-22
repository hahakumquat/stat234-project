#!/bin/bash
#
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -t 5-00:00
#SBATCH --mem=4096
#SBATCH -o mountaincar_random_memory.out
#SBATCH -e mountaincar_random_memory.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rouyang@college.harvard.edu

python server_main.py MountainCarGame NoTraining Random
