import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

class NoTraining(nn.Module):

    def __init__(self, env, **kwargs):
        
        self.env = env
        self.train_counter = 0
        self.batch_size = -1
        self.target_update = -1
        self.gamma = -1
        self.learning_rate = -1
        self.lr_annealer = None
        self.optim_name = 'None'
        self.loss_name = 'None'
        self.regularization = -1

    def forward(self, state_batch):
        result = Variable(FloatTensor(np.ones((len(state_batch), self.env.action_space.n))))
        return result

    def train_model(self, memory):
        self.train_counter += 1
        return -1

