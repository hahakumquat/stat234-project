import numpy as np

import torch
from torch.autograd import Variable

class EpsilonGreedy():

    def __init__(self, models, env, epsilon_start=1, epsilon_end=0.05, epsilon_decay=200):
        self.eps_start = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_decay
        self.env = env
        try:
            iter(models)
        except TypeError:
            # models is not iterable
            self.models = [models]
        else:
            self.models = models
        self.steps_done = 0

    def select_action(self, state):
        sample = np.random.random()
        threshold = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1 * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > threshold:
            values = torch.zeros((1, self.env.action_space.n))
            for model in self.models:
                values += model.forward(Variable(state, volatile=True)).data
            return values.max(1)[1].view(1, 1)[0, 0]
        else:
            return self.env.action_space.sample()
        
