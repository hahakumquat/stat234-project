#!/bin/bash
#
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -t 5-00:00
#SBATCH -p general
#SBATCH --mem=8192
#SBATCH -o cartpole_dqn.out
#SBATCH -e cartpole_dqn.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michaelge@college.harvard.edu

python server_main.py CartPoleGame DQN_GS EpsilonGreedy