import sys
import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torchvision.transforms as T
from PIL import Image
import pickle
import argparse

import time
import datetime

timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

parser = argparse.ArgumentParser(description='Run RL simulation.')
parser.add_argument('-g', metavar='game', default='CartPoleGame', help='The game name.')
parser.add_argument('-m', metavar='model', default='DQN_GS', help='The model name.')
parser.add_argument('-a', metavar='agent', default='EpsilonGreedy', help='The agent name.')
parser.add_argument('-e', metavar='episodes', type=int, default=1000, help='Number of episodes.')
parser.add_argument('--server',  action='store_true', help='Creates a fake window for server-side running.')
parser.add_argument('--base_network',  action='store_true', help='Starts training from a network with pre-trained weights.')
parser.add_argument('--nreplay', metavar='replay_size', type=int, default=10000, help='Size of replay memory.')

args = parser.parse_args()
if args.server:
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(400, 600))
    display.start()

# first change the cwd to the script path
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)

sys.path.append('games')
sys.path.append('models')
sys.path.append('agents')
sys.path.append('utils')

from ReplayMemory import ReplayMemory, Transition
from Logger import Logger

# models
from DQN_GS import DQNGS
from NoTraining import NoTraining

# agents
from EpsilonGreedy import EpsilonGreedy
from Random import Random

# games
from CartPoleGame import CartPoleGame
from AcrobotGame import AcrobotGame
from MountainCarGame import MountainCarGame
from CartPoleCroppedGame import CartPoleCroppedGame

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

memory = ReplayMemory(args.nreplay)
total_rewards = []
episode_durations = []
frame_skip = 3
update_frequency = 4
BATCH_SIZE = 128

game = None
model = None
agent = None
sample_states = None

game_name = args.g
model_name = args.m
agent_name = args.a
num_episodes = args.e

if game_name == 'CartPoleGame':
    game = CartPoleGame()
elif game_name == 'AcrobotGame':
    game = AcrobotGame()
elif game_name == 'MountainCarGame':
    game = MountainCarGame()
elif game_name == 'CartPoleCroppedGame':
    game = CartPoleCroppedGame()
else:
    raise Exception('Game does not exist. Ex: For CartPoleGame.py, use CartPoleGame')

if model_name == 'NoTraining':
    model = NoTraining(game.env)
elif model_name == 'DQN':
    model = DQN(game.env)
elif model_name == 'DQN_GS':
    model = DQNGS(game.env)
else:
    raise Exception('Model does not exist. Ex: For DQN.py, use DQN')
if use_cuda:
    model.cuda()
    print('Using CUDA.')

if agent_name == 'EpsilonGreedy':
    agent = EpsilonGreedy(model, game.env)
elif agent_name == 'Random':
    agent = Random(model, game.env)
else:
    raise Exception('Agent does not exist. Ex: For EpsilonGreedy.py, use EpsilonGreedy')

filename = game.file_prefix + model_name + '_' + agent_name
reward_log = Logger('results/' + game_name + '/' + filename + '_rewards_' + timestamp + '.csv')
duration_log = Logger('results/' + game_name + '/' + filename + '_durations_' + timestamp + '.csv')
loss_log = Logger('results/' + game_name + '/' + filename + '_losses_' + timestamp + '.csv')
Q_log = Logger('results/' + game_name + '/' + filename + '_sample_Q_' + timestamp + '.csv')

# get sample states to compute Q function instead of (in addition to) average reward
if model_name != 'NoTraining':
    replay_memory_file = 'data/sample_states/' + game.file_prefix + 'NoTraining_Random_memory.pkl'
    if os.path.exists(replay_memory_file):
        with open(replay_memory_file, 'rb') as f:
            sample_states = pickle.load(f)
        sample_states = Variable(torch.cat(sample_states))
        print('Loaded in sample states.')

if args.base_network and model_name != 'NoTraining':
    network_file_to_load = 'data/networks/' + game.file_prefix + 'DQN_GS_Random_network.pt'
    if os.path.exists(network_file_to_load):
        model.load_state_dict(torch.load(network_file_to_load))
        print('Loaded pre-trained network.')

def main(batch_sz, num_episodes):
    
    for i_episode in range(num_episodes):
        game.env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen
        state_info = game.env.state
        total_reward = 0
        t = 0
        done = False
        while not done:
            # Select and perform an action
            action = agent.select_action(state)
            frame_skip_reward = 0
            for i_frame_skip in range(frame_skip):
                _, reward, done, _ =  game.env.step(action[0, 0])
                frame_skip_reward += reward
                if done:
                    break
                t += 1

            total_reward += frame_skip_reward
            frame_skip_reward = FloatTensor([frame_skip_reward])

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen()
            if not done:
                next_state = current_screen - last_screen

                # get OpenAI Gym's 4 state elements
                next_state_info = game.env.state
            else:
                next_state = None
                next_state_info = None

            # Store the transition in memory
            memory.push(state, action, frame_skip_reward, next_state, state_info, next_state_info)

            # Move to the next state
            state = next_state
            state_info = next_state_info

            # Perform one step of the optimization (on the target network)
            if len(memory) >= BATCH_SIZE:

                # only train every frame_skip * update_frequency time steps, 
                # i.e., only train after update_frequency different actions 
                # have been selected. This speeds up training. See DQN paper.
                if t % (frame_skip * update_frequency) == 0:
                    loss_log.log(model.train(memory))

            if done:
                print('Done! Duration:', t + 1)
                total_rewards.append(total_reward)
                reward_log.log(total_reward)
                episode_durations.append(t + 1)
                duration_log.log(t + 1)
                if sample_states is not None:
                    Q_log.log(model.compute_sample_Q(sample_states))
                break
            
def get_screen():
    if model_name == 'DQN':
        screen = np.array(game.env.render(mode='rgb_array')).transpose((2, 0, 1))
    elif model_name == 'DQN_GS':
        screen = np.expand_dims(Image.fromarray(game.env.render(mode='rgb_array')).convert('L'), axis=2).transpose((2, 0, 1))
    else:
        screen = np.expand_dims(Image.fromarray(game.env.render(mode='rgb_array')).convert('L'), axis=2).transpose((2, 0, 1))

    screen = game.modify_screen(screen)

    screen = np.ascontiguousarray(screen, dtype=np.float32)
    screen /= 255
    screen = torch.from_numpy(screen)

    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).type(FloatTensor)

def resize(screen):
    rsz = T.Compose([T.ToPILImage(),
            T.Resize((80, 80), interpolation=Image.CUBIC),
            T.ToTensor()])
    return rsz(screen)

try:
    main(BATCH_SIZE, num_episodes)
except KeyboardInterrupt:
    print('Detected KeyboardInterrupt. ')
finally:
    game.env.close()
    if model_name != 'NoTraining' and agent_name == 'Random': # then we actually trained a DQN
        base_network_filename = 'results/' + game_name + '/' + filename + '_network.pt'
        # don't think we're actually going to save this until the end
        torch.save(model.state_dict(), base_network_filename)
        print('Saved random policy network.')

        # # Later to restore and evaluate:
        # model = DQNGS(game.env)
        # model.load_state_dict(torch.load(pickle_filename))
        # model.eval()
        pass
    if agent_name == 'Random' and model_name == 'NoTraining': # it was random
        pickle_filename = 'results/' + game_name + '/' + filename + '_memory.pkl'
        if os.path.exists(pickle_filename):
            os.remove(pickle_filename)
        with open(pickle_filename, 'wb') as f:
            sample_states = memory.sample(BATCH_SIZE)
            sample_states = [sample_state.state for sample_state in sample_states]
            pickle.dump(sample_states, f)
        print('Saved ReplayMemory sample states.')
    else:
        pass
