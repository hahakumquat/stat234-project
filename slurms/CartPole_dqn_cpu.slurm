#!/bin/bash
#
#SBATCH -N 1
#SBATCH -t 5-00:00
#SBATCH -p general
#SBATCH --mem=8192
#SBATCH -o cartpole_dqn_cpu.out
#SBATCH -e cartpole_dqn_cpu.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michaelge@college.harvard.edu, rouyang@college.harvard.edu

cd ~/stat234-project/ && python main.py -g CartPoleGame -m DQN_GS -a EpsilonGreedy --server
