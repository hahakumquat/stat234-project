import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from Logger import Logger
from ReplayMemory import Transition

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

class DQNGS(nn.Module):

    def __init__(self, env, batch_sz=128, lr=0.1, gamma=0.99):
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

        self.lr_annealer = lambda epoch: max(np.exp(-epoch / 500 - 2.3), 0.0005)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, 
                     lr_lambda=self.lr_annealer)
        
        self.loss_function = nn.MSELoss()
        self.train_counter = 0

    def forward(self, state_batch):
        state_batch = self.relu1(self.bn1(self.conv1(state_batch)))
        state_batch = self.relu2(self.bn2(self.conv2(state_batch)))
        state_batch = self.mp(self.relu3(self.bn3(self.conv3(state_batch))))
        state_batch = state_batch.view(len(state_batch), -1)
        result = self.out_layer(state_batch)
        return result

    def train_model(self, memory, target_network):
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

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.forward(state_batch)
        state_action_values = state_action_values.gather(1, action_batch)

        # Compute max_{a'} Q(s_{t+1}, a') for all next states.
        next_state_values = Variable(torch.zeros(len(state_batch)).type(FloatTensor))
        if target_network is not None:
            next_state_values[non_final_mask] = target_network.forward(non_final_next_states).max(1)[0]
        else:
            next_state_values[non_final_mask] = self.forward(non_final_next_states).max(1)[0]
        
        # Now, we don't want to mess up the loss with a volatile flag, so let's
        # clear it. After this, we'll just end up with a Variable that has
        # requires_grad=False
        # next_state_values.volatile = False

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # The below line can replace next_state_values.volatile = False. See PyTorch DQN tutorial.
        expected_state_action_values = Variable(expected_state_action_values.data)

        # Compute loss
        loss = self.loss_function(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.scheduler.step()
        self.train_counter += 1
        return loss.data[0] / len(state_action_values)

    def compute_sample_Q(self, sample_states):
        res = self.forward(sample_states).max(1)[0].mean(0).data[0]
        print(res.shape)
        return float(res)