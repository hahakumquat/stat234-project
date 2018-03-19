import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from Logger import Logger

class NoTraining(nn.Module):

    def __init__(self, env, batch_size=128):
        
        self.env = env
        self.batch_size = batch_size

    def forward(self, state_batch):
        return torch.FloatTensor(np.ones((self.batch_size, self.env.action_space.n)))

    def train(self, state_batch, action_batch, reward_batch, next_state_values):

        return

