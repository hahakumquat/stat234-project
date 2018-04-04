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

timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%d-%m-%Y__%H_%M_%S')

parser = argparse.ArgumentParser(description='Run RL simulation.')
parser.add_argument('-g', metavar='game', default='CartPole-v0', help='The game name.')
parser.add_argument('-m', metavar='model', default='DQN_GS', help='The model name.')
parser.add_argument('-a', metavar='agent', default='EpsilonGreedy', help='The agent name.')
parser.add_argument('-e', metavar='ntrains', type=int, default=50000, help='Number of trains.')
parser.add_argument('--server', action='store_true', help='Creates a fake window for server-side running.')
parser.add_argument('--base_network', action='store_true', help='Starts training from a network with pre-trained weights.')
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

# utils
from ReplayMemory import ReplayMemory, Transition
from Logger import Logger

# models
from DQN_GS import DQNGS
from DDQN_GS import DDQNGS
from NoTraining import NoTraining

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
BATCH_SIZE = 128
target_update = 1000

game = None
model = None
agent = None
sample_states = None
enable_target_network = False

game_name = args.g
model_name = args.m
agent_name = args.a
num_trains = args.e

if game_name == 'CartPoleCroppedGame':
    game = CartPoleCroppedGame()
else:
    game = Game(game_name)

if model_name == 'NoTraining':
    model = NoTraining(game.env)    
elif model_name == 'DQN_GS':
    model = DQNGS(game.env, use_target_network=enable_target_network, target_update=target_update)
elif model_name == 'DDQN_GS':
    model = DDQNGS(game.env)
else:
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
        Q_log = Logger(filename + '_sample_Q_' + cuda_label + '_' + timestamp + '.csv')

if args.base_network and model_name != 'NoTraining':
    network_file_to_load = 'data/networks/' + game.file_prefix + 'DQN_GS_Random_network_' + cuda_label + '.pt'
    if os.path.exists(network_file_to_load):
        model.load_state_dict(torch.load(network_file_to_load))
        print('Loaded pre-trained network.', flush=True)

notes_log = Logger(filename + '_notes_' + cuda_label + '_' + timestamp + '.txt')

notes_log.log('GAME PARAMETERS')
notes_log.log('Game: ' + game_name)
notes_log.log('Model: ' + model_name)
notes_log.log('Agent: ' + agent_name)
notes_log.log('N Trains: ' + str(num_trains))
notes_log.log('Processing: ' + cuda_label)
notes_log.log('Has target network: ' + str(model.use_target_network))
notes_log.log('Frame skip: ' + str(frame_skip))
notes_log.log('Update frequency: ' + str(update_frequency))
notes_log.log('NETWORK PARAMETERS')
notes_log.log('batch size: ' + str(model.batch_size))
notes_log.log('gamma: ' + str(model.gamma))
notes_log.log('Initial learning rate: ' + str(model.learning_rate))
notes_log.log('learning rate annealing: ' + str(model.lr_annealer is not None))
notes_log.log('Optimizer: ' + model.optim_name)
notes_log.log('Loss Function: ' + model.loss_name)
notes_log.log('weight decay: ' + str(model.regularization))
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
        update_target = False
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

            if done or t > 20000:
                reward_log.log(total_reward)
                duration_log.log(t + 1)
                print(t + 1)
                if sample_states is not None:
                    Q_log.log(model.compute_sample_Q(sample_states))
                break
        # num_episodes += 1
            
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
            T.ToTensor()])
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
