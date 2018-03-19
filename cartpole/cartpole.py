import sys
import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torchvision.transforms as T
from PIL import Image

from itertools import count
# from copy import deepcopy

# first change the cwd to the script path
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)

# append the relative location you want to import from
sys.path.append('../utils')

from ReplayMemory import ReplayMemory, Transition
from Logger import Logger

# models
from DQN import DQN
from DQN_GS import DQNGS

# agents
from EpsilonGreedy import EpsilonGreedy
from Random import Random

memory = ReplayMemory(10000)
episode_durations = []
duration_log = Logger('durations.csv')
total_rewards = []
reward_log = Logger('rewards.csv')
frame_skip = 2

env = gym.make('CartPole-v0').unwrapped
screen_width = 600
model = None
agent = None

if len(sys.argv) == 3:
    model_name = sys.argv[1]
    agent_name = sys.argv[2]
    if model_name == 'DQN':
        model = DQN(env)
    elif model_name == 'DQN_GS':
        model = DQNGS(env)
    else:
        raise Exception('Model does not exist. Ex: For DQN.py, use DQN')
    if agent_name == 'EpsilonGreedy':
        agent = EpsilonGreedy(model, env)
    elif agent_name == 'Random':
        agent = Random(model, env)
    else:
        raise Exception('Agent does not exist. Ex: For EpsilonGreedy.py, use EpsilonGreedy')
else:
    raise Exception('Usage: python main.py <model_name> <agent_name>')

def main(batch_sz, num_episodes):
    for i_episode in range(num_episodes):

        # Initialize the environment and state
        env.reset()
        last_screen = get_screen(env)
        current_screen = get_screen(env)
        state = current_screen - last_screen
        state_info = env.state
        total_reward = 0
        for t in count():
            # Select and perform an action
            action = agent.select_action(state)
            frame_skip_reward = 0
            for i_frame_skip in range(frame_skip):
                _, reward, done, _ =  env.step(action[0, 0])
                frame_skip_reward += reward
                if done:
                    break
            total_reward += frame_skip_reward
            frame_skip_reward = torch.FloatTensor([frame_skip_reward])

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(env)
            if not done:
                next_state = current_screen - last_screen

                # get OpenAI Gym's 4 state elements
                next_state_info = env.state
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
                transitions = memory.sample(BATCH_SIZE)
                # stackoverflow: 
                batch = Transition(*zip(*transitions))

                # Compute a mask of non-final states and concatenate the batch elements
                non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                      batch.next_state)))

                # We don't want to backprop through the expected action values and volatile
                # will save us on temporarily changing the model parameters'
                # requires_grad to False!
                non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)
                state_batch = Variable(torch.cat(batch.state))
                action_batch = Variable(torch.cat(batch.action))
                reward_batch = Variable(torch.cat(batch.reward))
                # Compute V(s_{t+1}) for all next states.
                next_state_values = Variable(torch.zeros(len(state_batch)).type(torch.FloatTensor))
                next_state_values[non_final_mask] = model.forward(non_final_next_states).max(1)[0]
                next_state_values.volatile = False

                # Now, we don't want to mess up the loss with a volatile flag, so let's
                # clear it. After this, we'll just end up with a Variable that has
                # requires_grad=False
                # next_state_values.volatile = False

                model.train(state_batch, action_batch, reward_batch, next_state_values)

            if done:
                episode_durations.append(t + 1)
                duration_log.log(t + 1)
                total_rewards.append(total_reward)
                reward_log.log(total_reward)
                break

def get_screen(env):
    if sys.argv[1] == 'DQN':
        screen = env.render(mode='rgb_array').tranpose((2, 0, 1))
    elif sys.argv[1] == 'DQN_GS':
        screen = np.expand_dims(Image.fromarray(env.render(mode='rgb_array')).convert('L'), axis=2).transpose((2, 0, 1))
    view_width = 160
    cart_location = get_cart_location(env)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)
    # Strip off the top and bottom of the screen
    screen = screen[:, 160:320, slice_range]

    # plt.imshow(screen[0], cmap='gray')
    # plt.savefig('screen.pdf')
    # plt.close()
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    # print(resize(screen).unsqueeze(0).type(torch.FloatTensor).shape)
    return resize(screen).unsqueeze(0).type(torch.FloatTensor)

def get_cart_location(env):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)

def resize(screen):
    rsz = T.Compose([T.ToPILImage(),
            T.Resize((80, 80), interpolation=Image.CUBIC),
            T.ToTensor()])
    return rsz(screen)

BATCH_SIZE = 128
num_episodes = 1000
try:
    main(BATCH_SIZE, num_episodes)
except KeyboardInterrupt:
    print('Detected KeyboardInterrupt. ')
    # pickle the neural net
