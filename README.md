# stat234-project
OpenAI Gym

1. Download/Install Anaconda
2. Set up Conda environment
- ``conda env create -n boxcar python=3.5``
- ``source activate boxcar``

3. Install packages
- ``conda install -c https://conda.anaconda.org/kne pybox2d``
- ``pip install gym``
- ``conda install pytorch-cpu torchvision -c pytorch`` (but better to build pytorch and torchvision from source)
- ``conda install matplotlib``
4. Fix errors if unmerged into master OpenAI gym branch
- https://github.com/openai/gym/pull/930
- In gym/gym/envs/classic_control/cartpole.py, change ``self.observation_space = spaces.Box(-high, high)`` to ``self.observation_space = spaces.Box(-high, high, dtype=np.float32)`` to suppress logger warning
5. Run ``main.py``
- ``python main.py -h`` for command-line argument help
- ``python main.py -g CartPole-v0 -m DQN_GS -a EpsilonGreedy -e 1000 --nreplay 10000`` for training grayscale DQN
- ``python main.py -g CartPole-v0 -m NoTraining -a Random -e 1000 --nreplay 10000`` for random policy

## Baselines over 1000 trials
See ``data`` folder
- Acrobot Random: average duration 599.73, average reward -598.73
- CartPole Random: average duration 18.441, average reward 18.441
- MountainCar Random: average duration 2717.77, average reward -2717.77
