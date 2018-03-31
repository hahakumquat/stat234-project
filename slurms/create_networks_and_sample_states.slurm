#!/bin/bash
#
#SBATCH -N 1
#SBATCH -t 5-00:00
#SBATCH -p general
#SBATCH --mem=16000
#SBATCH -o create.out
#SBATCH -e create.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michaelge@college.harvard.edu, rouyang@college.harvard.edu

# run each game for 1000 episodes
# cd ~/stat234-project/ && python main.py -g CartPole-v0 -m DQN_GS -a Random -e 10000 --server
# cp ~/stat234-project/results/CartPoleGame/cartpole_DQN_GS_Random_network.pt ~/stat234-project/data/networks
# cp ~/stat234-project/results/CartPoleGame/cartpole_NoTraining_Random_memory.pkl ~/stat234-project/data/sample_states
# cd ~/stat234-project/ && python main.py -g CartPole-v0 -e 32 --base_network --server

cd ~/stat234-project/ && python main.py -g Acrobot-v1 -m DQN_GS -a Random -e 10000 --server
cp ~/stat234-project/results/AcrobotGame/acrobot_DQN_GS_Random_network.pt ~/stat234-project/data/networks
cp ~/stat234-project/results/AcrobotGame/acrobot_NoTraining_Random_memory.pkl ~/stat234-project/data/sample_states
cd ~/stat234-project/ && python main.py -g Acrobot-v1 -e 32 --base_network --server

cd ~/stat234-project/ && python main.py -g MountainCar-v0 -m DQN_GS -a Random -e 10000 --server
cp ~/stat234-project/results/MountainCarGame/mountaincar_DQN_GS_Random_network.pt ~/stat234-project/data/networks
cp ~/stat234-project/results/MountainCarGame/mountaincar_NoTraining_Random_memory.pkl ~/stat234-project/data/sample_states
cd ~/stat234-project/ && python main.py -g MountainCar-v0 -e 32 --base_network --server

# check if Q-network values start small
