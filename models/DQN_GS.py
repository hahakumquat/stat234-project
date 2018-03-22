import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

from Logger import Logger
from ReplayMemory import Transition

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

class DQNGS(nn.Module):

    def __init__(self, env, batch_sz=128, lr=0.001, gamma=0.99):
        super(DQNGS, self).__init__()

        ## DQN architecture
        self.conv1 = nn.Conv2d(1, 8, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.LeakyReLU(0.01)

        self.conv3 = nn.Conv2d(16, 16, kernel_size=4, stride=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.LeakyReLU(0.01)
        
        self.mp = nn.MaxPool2d(2)
        
        self.out_layer = nn.Linear(64, env.action_space.n)

        self.env = env
        self.batch_size = batch_sz
        self.learning_rate = lr
        self.gamma = gamma
        
        self.optimizer = optim.RMSprop(self.parameters(),
                                       lr=self.learning_rate)
        self.loss_function = F.smooth_l1_loss
        self.losses = []

    def forward(self, state_batch):
        state_batch = self.relu1(self.bn1(self.conv1(state_batch)))
        state_batch = self.relu2(self.bn2(self.conv2(state_batch)))
        state_batch = self.mp(self.relu3(self.bn3(self.conv3(state_batch))))
        state_batch = state_batch.view(len(state_batch), -1)
        result = self.out_layer(state_batch)
        return result

    def train(self, memory):
        transitions = memory.sample(self.batch_size)
        # stackoverflow: 
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)))

        # We don't want to backprop through the expected action values and volatile
        # will save us on temporarily changing the model parameters'
        # requires_grad to False!
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), 
                                         volatile=True)
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(len(state_batch)).type(FloatTensor))
        next_state_values[non_final_mask] = self.forward(non_final_next_states).max(1)[0]

        # Now, we don't want to mess up the loss with a volatile flag, so let's
        # clear it. After this, we'll just end up with a Variable that has
        # requires_grad=False
        next_state_values.volatile = False

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.forward(state_batch)
        state_action_values = state_action_values.gather(1, action_batch)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        self.losses.append(loss.data / len(state_action_values))     

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.data[0] / len(state_action_values)

    def compute_sample_Q(self, sample_states):
        return self.forward(sample_states).max(1)[0].mean(0).data[0]
