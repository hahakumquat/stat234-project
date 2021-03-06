#!/usr/bin/env python3.5

import argparse
import datetime
import numpy as np
import os
import pickle
from PIL import Image
import sys
import time

import gym
import torch
from torch.autograd import Variable
import torchvision.transforms as T
import random

parser = argparse.ArgumentParser(description='Run reinforcement learning simulation.')
parser.add_argument('-g', metavar='game_name', default='CartPole-v0', help='One of 3 classic control games (CartPole-v0, Acrobot-v1, MountainCar-v0). Default is CartPole-v0.')
parser.add_argument('-m', metavar='model_name', default='DQN-GS', help='The model type (NoTraining, DQN-GS, DDQN-GS, DQN-PCA, DDQN-PCA). Default is DQN-GS.')
parser.add_argument('-a', metavar='agent_name', default='EpsilonGreedy', help='The agent type (EpsilonGreedy, Random). Default is EpsilonGreedy.')
parser.add_argument('-e', metavar='num_trains', type=int, default=50000, help='Number of minibatch trains. Default is 50000.')
parser.add_argument('--server', action='store_true', help='Creates a fake window for server-side running. Default is False.')
parser.add_argument('--base_network', action='store_true', help='Starts training from a network with pre-trained weights. Default is False.')
parser.add_argument('--nreplay', metavar='replay_size', type=int, default=10000, help='Size of replay memory. Default is 10000.')
parser.add_argument('--target', metavar='target_update', type=int, default=0, help='Target network update. Default is 0.')


# Neural Network Parameters
parser.add_argument('--linears', metavar='linears', type=str, default="128_64", help='Layer sizes, separated by underscores. Default is 128_64.')
parser.add_argument('--lr', metavar='learning_rate', type=float, default=0.001, help='Learning rate. Default is 0.001.')
parser.add_argument('--batch', metavar='batch_sizes', type=int, default=128, help='Batch size. Default is 128.')
parser.add_argument('--anneal', action='store_true', help='Turns on learning rate annealing. Default is False.')
parser.add_argument('--noanneal', action='store_true', help='Turns off learning rate annealing.')
parser.add_argument('--loss', metavar='loss', type=str, default='Huber', help='Loss function. Default is Huber.')
parser.add_argument('--regularization', metavar='regularization', type=float, default=0.1, help='L2 regularization. Default is 0.1.')

args = parser.parse_args()
if args.server:
    print('Server enabled.', flush=True)
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(400, 600))
    display.start()
    timestamp = str(random.random())[2:]
else:
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%d-%m-%Y__%H_%M_%S')
print(timestamp, flush=True)

# first change the cwd to the script path
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)

sys.path.append('games')
sys.path.append('models')
sys.path.append('agents')
sys.path.append('utils')

# utils
from ReplayMemory import ReplayMemory, Transition
from Logger import Logger
from PCA import PCA

# agents
from EpsilonGreedy import EpsilonGreedy
from Random import Random

# games
from Game import Game
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
BATCH_SIZE = args.batch
target_update = args.target

game = None
model = None
agent = None
sample_states = None

game_name = args.g
model_name = args.m
agent_name = args.a
num_trains = args.e

linears = [int(x) for x in args.linears.split('_')]
lr = args.lr
batch = args.batch
anneal = args.anneal
loss_function = args.loss
reg = args.regularization

if game_name == 'CartPoleCroppedGame':
    game = CartPoleCroppedGame()
else:
    game = Game(game_name)

model_name = model_name.replace('_', '-')
model_parameters = {'env': game.env,
                    'batch_sz': batch,
                    'lr': lr,
                    'gamma': 0.99,
                    'regularization': reg,
                    'target_update': 0 if 'DDQN' in model_name else target_update,
                    'anneal': anneal,
                    'loss': loss_function
                    }
if 'PCA' in model_name:
    model_parameters['pca_path'] = 'data/states/' + game.file_prefix + 'PCA.pkl'
if 'DDQ' in model_name:
    model_parameters['model'] = model_name.replace('DDQ', 'DQ')
if 'DQN-PCA' in model_name:
    model_parameters['linears'] = linears

try:
    if 'DDQ' in model_name:
        from DDQN import DDQN
        model = DDQN(**model_parameters)
    else:
        module_name = __import__(model_name.replace('-', '_'))
        model = getattr(module_name, model_name.replace('-', ''))(**model_parameters)
except KeyError:
    raise Exception('Model does not exist. Ex: For DQN.py, use DQN')    
if use_cuda:
    model.cuda()
    print('Using CUDA.', flush=True)
    cuda_label = 'gpu'
else:
    print('No CUDA. Using CPU.', flush=True)
    cuda_label = 'cpu'

if agent_name == 'EpsilonGreedy':
    agent = EpsilonGreedy(model, game.env)
elif agent_name == 'Random':
    agent = Random(model, game.env)
else:
    raise Exception('Agent does not exist. Ex: For EpsilonGreedy.py, use EpsilonGreedy')

filename = 'results/' + game_name + '/' + timestamp + '/' + game.file_prefix + model_name + '_' + agent_name
print('filename: ', filename, flush=True)
reward_log = Logger(filename + '_rewards_' + cuda_label + '.csv')
duration_log = Logger(filename + '_durations_' + cuda_label + '.csv')
if model_name != 'NoTraining':
    loss_log = Logger(filename + '_losses_' + cuda_label + '.csv')

# get sample states to compute Q function instead of (in addition to) average reward
if model_name != 'NoTraining':
    replay_memory_file = 'data/sample_states/' + game.file_prefix + 'NoTraining_Random_memory_' + cuda_label + '.pkl'
    if os.path.exists(replay_memory_file):
        with open(replay_memory_file, 'rb') as f:
            sample_states = pickle.load(f)
        sample_states = Variable(torch.cat(sample_states))
        print('Loaded in sample states.', flush=True)
        Q_log = Logger(filename + '_sample_Q_' + cuda_label + '.csv')

if args.base_network and model_name != 'NoTraining':
    network_file_to_load = 'data/networks/' + game.file_prefix + 'DQN-GS_Random_network_' + cuda_label + '.pt'
    if os.path.exists(network_file_to_load):
        model.load_state_dict(torch.load(network_file_to_load))
        print('Loaded pre-trained network.', flush=True)

notes_log = Logger(filename + '_notes_' + cuda_label + '.txt')

notes_log.log('GAME PARAMETERS')
notes_log.log('game: ' + game_name)
notes_log.log('model: ' + model_name)
notes_log.log('agent: ' + agent_name)
notes_log.log('n_trains: ' + str(num_trains))
notes_log.log('processing: ' + cuda_label)
notes_log.log('target_update: ' + str(model.target_update))
notes_log.log('frame_skip: ' + str(frame_skip))
notes_log.log('update_frequency: ' + str(update_frequency))
notes_log.log('NETWORK PARAMETERS')
notes_log.log('batch_size: ' + str(model.batch_size))
notes_log.log('gamma: ' + str(model.gamma))
notes_log.log('initial_learning_rate: ' + str(model.learning_rate))
notes_log.log('annealing: ' + str(model.lr_annealer is not None))
notes_log.log('optimizer: ' + model.optim_name)
notes_log.log('loss_function: ' + model.loss_name)
notes_log.log('weight_decay: ' + str(model.regularization))
notes_log.log('layer_sizes: ' + ('Default' if 'DQN-PCA' not in model_name else str(linears).replace(', ', '~')))
notes_log.log('num_parameters: ' + str(model.total_parameters))
notes_log.close()
        
def main(batch_sz, num_trains):
    # num_episodes = 0
    update_frequency_counter = 0
    while model.train_counter < num_trains:
        game.env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen
        total_reward = 0
        t = 0
        done = False

        while not done:
            # Select and perform an action
            action = agent.select_action(state)
            frame_skip_reward = 0
            for i_frame_skip in range(frame_skip):
                _, reward, done, _ =  game.env.step(action)
                frame_skip_reward += reward
                t += 1
                if done:
                    break

            total_reward += frame_skip_reward
            frame_skip_reward = FloatTensor([frame_skip_reward])

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, LongTensor([[action]]), frame_skip_reward, next_state)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization
            if len(memory) >= BATCH_SIZE and model_name != 'NoTraining':
                # only train after update_frequency different actions 
                # have been selected. This speeds up training. See DQN paper.
                if update_frequency_counter % update_frequency == 0:
                    loss_log.log(model.train_model(memory))
                update_frequency_counter += 1

            if done or t > 10000:
                reward_log.log(total_reward)
                duration_log.log(t + 1)
                if not args.server:
                    print(t + 1)
                if sample_states is not None:
                    Q_log.log(model.compute_sample_Q(sample_states))
                break
        # num_episodes += 1
            
def get_screen():
    if model_name == 'DQN':
        screen = np.array(game.env.render(mode='rgb_array')).transpose((2, 0, 1))
    elif model_name == 'DQN-GS':
        screen = np.expand_dims(Image.fromarray(game.env.render(mode='rgb_array')).convert('L'), axis=2).transpose((2, 0, 1))
    else:
        screen = np.expand_dims(Image.fromarray(game.env.render(mode='rgb_array')).convert('L'), axis=2).transpose((2, 0, 1))

    screen = game.modify_screen(screen)

    screen = np.ascontiguousarray(screen, dtype=np.float32)
    screen /= 255
    screen = torch.from_numpy(screen)

    # # save grayscale, processed image
    # plt.imshow(resize(screen).numpy()[0], cmap='gray')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig(game.file_prefix + 'processed.pdf', bbox_inches='tight')

    # # save original render image
    # plt.imshow(game.env.render(mode='rgb_array'))
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig(game.file_prefix + 'original.pdf', bbox_inches='tight')

    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).type(FloatTensor)

def resize(screen):
    rsz = T.Compose([T.ToPILImage(),
                     T.Resize((80, 80), interpolation=Image.CUBIC),
                     T.ToTensor()
                    ])
    return rsz(screen)

try:
    main(BATCH_SIZE, num_trains)
except KeyboardInterrupt:
    print('Detected KeyboardInterrupt. ', flush=True)
finally:
    game.env.close()
    if model_name != 'NoTraining' and agent_name == 'Random': # then we actually trained a DQN
        base_network_filename = filename + '_network_' + cuda_label + '.pt'
        torch.save(model.state_dict(), base_network_filename)
        print('Saved random policy network.', flush=True)

        # # Later to restore and evaluate:
        # model = DQNGS(game.env)
        # model.load_state_dict(torch.load(pickle_filename))
        # model.eval()
    if agent_name == 'Random':# and model_name == 'NoTraining': # it was random
        pickle_filename = 'results/' + game_name + '/' + game.file_prefix + 'NoTraining_Random_memory_' + cuda_label + '.pkl'
        if os.path.exists(pickle_filename):
            os.remove(pickle_filename)
        with open(pickle_filename, 'wb') as f:
            sample_states = memory.sample(BATCH_SIZE)
            sample_states = [sample_state.state for sample_state in sample_states]
            pickle.dump(sample_states, f)
        print('Saved ReplayMemory sample states.', flush=True)
