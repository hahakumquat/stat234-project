import sys
import os
import gym
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

# models
from DQN import DQN

# agents

from itertools import count
# from copy import deepcopy
# from PIL import Image

#first change the cwd to the script path
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)

#append the relative location you want to import from
sys.path.append("../utils")

#import your module stored in '../common'
from ReplayMemory import ReplayMemory, Transition

memory = ReplayMemory(1000)

env = gym.make('CartPole-v0').unwrapped
model = None
agent = None

if len(sys.argv) == 3:
    model_name = sys.argv[1]
    agent_name = sys.argv[2]
    if model_name == 'DQN':
        model = DQN(env)
    else:
        raise Exception('Model does not exist. Ex: For DQN.py, use DQN')
    if agent_name == 'EpsilonGreedy':
        agent = EpsilonGreedy(model)
    else:
        print("TODO")
        # raise Exception('Agent does not exist. Ex: For EpsilonGreedy.py, use EpsilonGreedy')
else:
    raise Exception("Usage: python main.py <model_name> <agent_name>")

BATCH_SIZE = 128
num_episodes = 1000
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    last_screen = env.render(mode='rgb_array')
    current_screen = env.render(mode='rgb_array')
    state = current_screen - last_screen
    for t in count():
        # Select and perform an action
        # action = agent.select_action(state)
        action = env.action_space.sample()
        _, reward, done, _ =  env.step(action) # env.step(action[0, 0])
        reward = torch.FloatTensor([reward])

        # Observe new state
        last_screen = current_screen
        current_screen = env.render(mode='rgb_array')
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        if len(memory) >= BATCH_SIZE:
            transitions = memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))
            model.train(transitions)
            
        if done:
            episode_durations.append(t + 1)
            # plot_durations()
            break

print('Complete')
