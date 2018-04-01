import gym
import numpy as np
import os
import pickle
import sys
import torch

# first change the cwd to the script path
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)

num_frames = sys.argv[2] if len(sys.argv) > 2 else 100

game_name = sys.argv[1] if len(sys.argv) > 1 else 'CartPole-v0'
env = gym.make(game_name).unwrapped
frame_skip = 3

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def get_screen():
    # we gotta do the grayscaled version
    screen = np.expand_dims(Image.fromarray(game.env.render(mode='rgb_array')).convert('L'), axis=2).transpose((2, 0, 1))

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

states = []
i_episode = 0
i_frame = 0
while i_frame < num_frames:
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    total_reward = 0
    t = 0
    done = False
    while not done:
        # Select and perform an action
        for i_frame_skip in range(frame_skip):
            _, _, done, _ =  env.step(env.action_space.sample())
            if done:
                break
            t += 1
            i_frame += 1

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        states.append(state)

        # Move to the next state
        state = next_state

        if done or t > 10000:
            break
    i_episode += 1

env.close()
pickle_filename = '../data/states/' + game_name.split('-')[0] + '_states.pkl'
if os.path.exists(pickle_filename):
    os.remove(pickle_filename)
with open(pickle_filename, 'wb') as f:
    pickle.dump(states, f)
print('Saved states for PCA.', flush=True)
