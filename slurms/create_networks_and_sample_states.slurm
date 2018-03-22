#!/bin/bash
#
#SBATCH -N 1
#SBATCH -t 5-00:00
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu
#SBATCH --mem=16000
#SBATCH -o create.out
#SBATCH -e create.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rouyang@college.harvard.edu

# run each game for 1000 episodes
cd ~/stat234-project/ && python main.py -g CartPoleGame -m DQN_GS -a Random -e 1000 --server
cp ~/stat234-project/results/CartPoleGame/cartpole_DQN_GS_Random_network.pt ~/stat234-project/data/networks
cp ~/stat234-project/results/CartPoleGame/cartpole_NoTraining_Random_memory.pkl ~/stat234-project/data/sample_states
cd ~/stat234-project/ && python main.py -g CartPoleGame -e 1000 --base_network --server

cd ~/stat234-project/ && python main.py -g AcrobotGame -m DQN_GS -a Random -e 1000 --server
cp ~/stat234-project/results/AcrobotGame/acrobot_DQN_GS_Random_network.pt ~/stat234-project/data/networks
cp ~/stat234-project/results/AcrobotGame/acrobot_NoTraining_Random_memory.pkl ~/stat234-project/data/sample_states
cd ~/stat234-project/ && python main.py -g AcrobotGame -e 1000 --base_network --server

cd ~/stat234-project/ && python main.py -g MountainCarGame -m DQN_GS -a Random -e 1000 --server
cp ~/stat234-project/results/MountainCarGame/mountaincar_DQN_GS_Random_network.pt ~/stat234-project/data/networks
cp ~/stat234-project/results/MountainCarGame/mountaincar_NoTraining_Random_memory.pkl ~/stat234-project/data/sample_states
cd ~/stat234-project/ && python main.py -g MountainCarGame -e 1000 --base_network --server

# check if Q-network values start small