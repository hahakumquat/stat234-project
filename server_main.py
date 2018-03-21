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

print("Modules loaded.")

from pyvirtualdisplay import Display
display = Display(visible=0, size=(400, 600))
display.start()

print("Display created.")

# from itertools import count
# from copy import deepcopy

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
from DQN import DQN
from DQN_GS import DQNGS
from NoTraining import NoTraining

# agents
from EpsilonGreedy import EpsilonGreedy
from Random import Random

# games
from CartPoleGame import CartPoleGame
from AcrobotGame import AcrobotGame
from MountainCarGame import MountainCarGame

memory = ReplayMemory(100000)
total_rewards = []
episode_durations = []
frame_skip = 4
update_frequency = 4

game = None
model = None
agent = None

if len(sys.argv) >= 2:
    game_name = sys.argv[1]
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

    if len(sys.argv) >= 4:
        model_name = sys.argv[2]
        agent_name = sys.argv[3]

        if model_name == 'DQN':
            model = DQN(game.env)
        elif model_name == 'DQN_GS':
            model = DQNGS(game.env)
        elif model_name == 'NoTraining':
            model = NoTraining(game.env)
        else:
            raise Exception('Model does not exist. Ex: For DQN.py, use DQN')
        if agent_name == 'EpsilonGreedy':
            agent = EpsilonGreedy(model, game.env)
        elif agent_name == 'Random':
            agent = Random(model, game.env)
        else:
            raise Exception('Agent does not exist. Ex: For EpsilonGreedy.py, use EpsilonGreedy')

        if len(sys.argv) >= 5:
            num_episodes = int(sys.argv[4])
        else:
            num_episodes = 10000
    else: 
        model_name = 'DQN_GS'
        model = DQNGS(game.env)
        agent_name = 'EpsilonGreedy'
        agent = EpsilonGreedy(model, game.env)
        num_episodes = 10000

else:
    raise Exception('Usage: python main.py game_name [model_name] [agent_name] [num_episodes]')

filename = game.file_prefix + model_name + '_' + agent_name
reward_log = Logger('results/' + game_name + '/' + filename + '_rewards.csv')
duration_log = Logger('results/' + game_name + '/' + filename + '_durations.csv')
loss_log = Logger('results/' + game_name + '/' + filename + '_losses.csv')

def main(batch_sz, num_episodes):
    for i_episode in range(num_episodes):

        # Initialize the environment and state
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
            frame_skip_reward = torch.FloatTensor([frame_skip_reward])

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
                break
            
def get_screen():
    if sys.argv[2] == 'DQN':
        screen = np.array(game.env.render(mode='rgb_array')).transpose((2, 0, 1))
    elif sys.argv[2] == 'DQN_GS':
        screen = np.expand_dims(Image.fromarray(game.env.render(mode='rgb_array')).convert('L'), axis=2).transpose((2, 0, 1))
    else:
        screen = np.expand_dims(Image.fromarray(game.env.render(mode='rgb_array')).convert('L'), axis=2).transpose((2, 0, 1))

    screen = game.modify_screen(screen)
    
    screen = np.ascontiguousarray(screen / 255, dtype=np.float32)
    screen = torch.from_numpy(screen)

    # Resize, and add a batch dimension (BCHW)
    # print(resize(screen).unsqueeze(0).type(torch.FloatTensor).shape)
    return resize(screen).unsqueeze(0).type(torch.FloatTensor)

def resize(screen):
    rsz = T.Compose([T.ToPILImage(),
            T.Resize((80, 80), interpolation=Image.CUBIC),
            T.ToTensor()])
    return rsz(screen)

BATCH_SIZE = 128
# num_episodes = 1000 # defined above with default 1000
try:
    main(BATCH_SIZE, num_episodes)
except KeyboardInterrupt:
    print('Detected KeyboardInterrupt. ')
finally:
    if model_name != 'NoTraining': # then we actually trained a DQN
        # pickle_filename = 'results/' + game_name + '/' + filename + '_network.pkl'
        # # don't think we're actually going to save this until the end
        # torch.save(model.state_dict(), pickle_filename)

        # # # Later to restore and evaluate:
        # # model = DQNGS(game.env)
        # # model.load_state_dict(torch.load(pickle_filename))
        # # model.eval()
        pass
    else: # it was random
        pickle_filename = 'results/' + game_name + '/' + filename + '_memory.pkl'
        if os.path.exists(pickle_filename):
            os.remove(pickle_filename)
        with open(pickle_filename, 'wb') as f:
            pickle.dump(memory, f)