#!/bin/bash
#
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 5-00:00
#SBATCH -p general
#SBATCH --mem=4096
#SBATCH -o acrobot_dqn.out
#SBATCH -e acrobot_dqn.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michaelge@college.harvard.edu

cd ..
python main.py -g AcrobotGame -m DQN_GS -a EpsilonGreedy -e 10000 --server
