import numpy as np
import random

import torch
from torch.autograd import Variable

class EpsilonGreedy():

    def __init__(self, model, env, epsilon_start=1, epsilon_end=0.05, epsilon_decay=200):
        self.eps_start = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_decay
        self.env = env
        self.model = model
        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        threshold = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1 * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > threshold:
            return self.model.forward(Variable(state, volatile=True)).data.max(1)[1].view(1, 1)
        else:
            return torch.LongTensor([[self.env.action_space.sample()]])
        
