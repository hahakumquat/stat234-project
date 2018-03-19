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
        result = Variable(torch.FloatTensor(np.ones((len(state_batch), self.env.action_space.n))))
        return result

    def train(self, memory):
        return -1

