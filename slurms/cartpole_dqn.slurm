#!/bin/bash
#
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 5-00:00
#SBATCH -p general
#SBATCH --mem=4096
#SBATCH -o cartpole_dqn.out
#SBATCH -e cartpole_dqn.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michaelge@college.harvard.edu

cd ~/stat234-project/ && python ../main.py -g CartPoleGame -m DQN_GS -a EpsilonGreedy -e 10000 --server
