import sys
import os
import gym
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

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
episode_durations = []

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

BATCH_SIZE = 10
num_episodes = 1000
for i_episode in range(num_episodes):

    # Initialize the environment and state
    env.reset()
    last_screen = env.render(mode='rgb_array')
    current_screen = env.render(mode='rgb_array')
    state = current_screen - last_screen
    state_info = env.state
    for t in count():
        # Select and perform an action
        # action = agent.select_action(state)
        action = env.action_space.sample()
        _, reward, done, _ =  env.step(action) # env.step(action[0, 0])

        # Observe new state
        last_screen = current_screen
        current_screen = env.render(mode='rgb_array')
        if not done:
            next_state = current_screen - last_screen

            # get OpenAI Gym's 4 state elements
            next_state_info = env.state
        else:
            next_state = None
            next_state_info = None

        # Store the transition in memory
        memory.push(state, action, reward, next_state, state_info, next_state_info)

        # Move to the next state
        state = next_state
        state_info = next_state_info

        # Perform one step of the optimization (on the target network)
        if len(memory) >= BATCH_SIZE:
            transitions = memory.sample(BATCH_SIZE)
            # stackoverflow: 
            batch = Transition(*zip(*transitions))
                
            # Compute a mask of non-final states and concatenate the batch elements
            non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                  batch.next_state)))

            # We don't want to backprop through the expected action values and volatile
            # will save us on temporarily changing the model parameters'
            # requires_grad to False!
            non_final_next_states = torch.from_numpy(np.array([s for s in batch.next_state if s is not None]))
            state_batch = torch.from_numpy(np.array(batch.state))
            action_batch = torch.FloatTensor(batch.action)
            reward_batch = torch.FloatTensor(batch.reward)
            # Compute V(s_{t+1}) for all next states.
            next_state_values = Variable(torch.zeros(len(state_batch)).type(torch.FloatTensor))
            next_state_values[non_final_mask] = model.forward(non_final_next_states).max(1)[0].data

            # Now, we don't want to mess up the loss with a volatile flag, so let's
            # clear it. After this, we'll just end up with a Variable that has
            # requires_grad=False
            # next_state_values.volatile = False

            model.train(state_batch, action_batch, reward_batch, next_state_values)
            
        if done:
            episode_durations.append(t + 1)
            # plot_durations()
            break

print('Complete')
