import numpy as np
import random
import torch

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

class Random():

    def __init__(self, model, env):
        self.env = env
        self.model = model
        self.steps_done = 0

    def select_action(self, state):
        return LongTensor([[self.env.action_space.sample()]])
