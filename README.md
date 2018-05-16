# stat234-project
(Double) Deep Q-Learning and PCA using OpenAI Gym

Paper located at https://github.com/rlouyang/stat234-writeup

## Setup
1. Download/install Anaconda

2. Clone this repository

3. ``conda env create -f environment.yml``

4. Fix errors if unmerged into master OpenAI gym branch
- https://github.com/openai/gym/pull/930
- In each of the game files in `gym/gym/envs/classic_control/`, add `dtype=np.float32` to each `spaces.Box()` initialization to suppress logger warning

5. Run ``main.py``
- ``python main.py -h`` for command-line argument help
- ``python main.py -g CartPole-v0 -m DQN_GS -a EpsilonGreedy -e 1000 --nreplay 10000`` for training grayscale DQN
- ``python main.py -g CartPole-v0 -m NoTraining -a Random -e 1000 --nreplay 10000`` for random policy

## Random baselines over 1000 trials
See ``data/random`` folder
- Acrobot Random: average duration 599.73, average reward -598.73
- CartPole Random: average duration 18.441, average reward 18.441
- MountainCar Random: average duration 2717.77, average reward -2717.77
