import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

class NoTraining(nn.Module):

    def __init__(self, env, batch_size=128):
        
        self.env = env
        self.batch_size = batch_size
        self.train_counter = 0

    def forward(self, state_batch):
        result = Variable(FloatTensor(np.ones((len(state_batch), self.env.action_space.n))))
        return result

    def train_model(self, memory, target_network):
        self.train_counter += 1
        return -1

