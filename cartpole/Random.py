import numpy as np
import random

class Random():

    def __init__(self, model, env):
        self.env = env
        self.model = model
        self.steps_done = 0

    def select_action(self, state):
        return torch.LongTensor([[self.env.action_space.sample()]])
