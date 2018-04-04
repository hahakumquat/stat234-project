import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from ReplayMemory import Transition

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

class DQNGS(nn.Module):

    def __init__(self, env, batch_sz=128, lr=0.01, gamma=0.99, regularization=0.0001, use_target_network=False, target_update=100):
        super(DQNGS, self).__init__()

        ## DQN architecture
        self.conv1 = nn.Conv2d(1, 8, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.LeakyReLU(0.0001)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.LeakyReLU(0.0001)

        self.conv3 = nn.Conv2d(16, 16, kernel_size=4, stride=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.LeakyReLU(0.0001)
        
        self.mp = nn.MaxPool2d(2)
        
        self.out_layer = nn.Linear(64, env.action_space.n)

        self.env = env
        self.batch_size = batch_sz
        
        self.learning_rate = lr
        self.lr_annealer = lambda epoch: max(np.exp(-epoch / 2000), 0.0005 / lr)
        
        self.gamma = gamma
        self.regularization = regularization
        
        self.optimizer = optim.RMSprop(self.parameters(),
                                       lr=self.learning_rate, weight_decay=regularization)
        self.optim_name = 'RMSprop'
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, 
                     lr_lambda=self.lr_annealer)
        
        self.loss_function = F.smooth_l1_loss
        self.loss_name = 'Huber Loss'
        self.train_counter = 0

        self.use_target_network = use_target_network
        if use_target_network:
            self.create_target_network()
            self.target_update = target_update

    def forward(self, state_batch):
        state_batch = self.relu1(self.bn1(self.conv1(state_batch)))
        state_batch = self.relu2(self.bn2(self.conv2(state_batch)))
        state_batch = self.mp(self.relu3(self.bn3(self.conv3(state_batch))))
        state_batch = state_batch.view(len(state_batch), -1)
        result = self.out_layer(state_batch)
        return result

    def train_model(self, memory):
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
        if self.use_target_network:
            next_state_values[non_final_mask] = self.target_network.forward(non_final_next_states).max(1)[0]
        else:
            next_state_values[non_final_mask] = self.forward(non_final_next_states).max(1)[0]
        
        # Now, we don't want to mess up the loss with a volatile flag, so let's
        # clear it. After this, we'll just end up with a Variable that has
        # requires_grad=False
        next_state_values.volatile = False

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # The below line can replace next_state_values.volatile = False. See PyTorch DQN tutorial.
        # expected_state_action_values = Variable(expected_state_action_values.data)

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

        if self.use_target_network and self.train_counter % self.target_update == 0:
            self.sync_target_network()

        return loss.data[0] / len(state_action_values)

    def compute_sample_Q(self, sample_states):
        res = self.forward(sample_states).data.max(1)[0].mean()
        return float(res)

    def create_target_network(self):
        self.target_network = DQNGS(env=env, batch_sz=self.batch_size, lr=self.learning_rate, gamma=self.gamma, regularization=self.regularization, use_target_network=False)
        self.target_network.load_state_dict(self.state_dict())
        self.target_network.eval() # can't train target_network again

    def cuda(self):
        super(DQNGS, self).cuda()
        if self.use_target_network:
            self.target_network.cuda()

    def load_state_dict(self, state_dict):
        super(DQNGS, self).load_state_dict(state_dict)
        if self.use_target_network:
            self.target_network.load_state_dict(state_dict)

    def sync_target_network(self):
        self.target_network.load_state_dict(self.state_dict())
