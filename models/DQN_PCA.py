import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle

from ReplayMemory import Transition

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

class DQNGS(nn.Module):

    def __init__(self, env, pca_path, batch_sz=128, lr=0.01, gamma=0.99, regularization=0.0001, target_update=0, anneal=False, loss='Huber'):
        super(DQNGS, self).__init__()

        with open(pca_path, 'rb') as f:
            self.pca = pickle.load(f)

        ## DQN architecture
        self.linear1 = nn.Linear(self.pca.n_pixels, 128)
        self.relu1 = nn.LeakyReLU(0.0001)

        self.linear2 = nn.Linear(128, 256)
        self.relu2 = nn.LeakyReLU(0.0001)

        self.linear3 = nn.Linear(256, 256)
        self.relu3 = nn.LeakyReLU(0.0001)
                
        self.out_layer = nn.Linear(256, env.action_space.n)

        self.env = env
        self.batch_size = batch_sz
        
        self.learning_rate = lr
        
        self.gamma = gamma
        self.regularization = regularization        
        
        self.loss_name = loss
        self.loss_function = None
        if loss == 'Huber':
            self.loss_function = F.smooth_l1_loss
        elif loss == 'MSE':
            self.loss_function = nn.MSELoss()
            
        self.train_counter = 0

        self.anneal = anneal
        
        self.use_target_network = (target_update > 0)
        self.target_update = target_update

        if self.use_target_network:
            self.create_target_network()
            
        self.optim_name = 'RMSprop'
        self.optimizer = optim.RMSprop(self.parameters(),
                                       lr=self.learning_rate,
                                       weight_decay=regularization)
        self.lr_annealer = None
        self.scheduler = None
        if self.anneal:
            self.lr_annealer = lambda epoch: max(np.exp(-epoch / 10000), 0.0005 / lr)
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, 
                                                         lr_lambda=self.lr_annealer)

    def forward(self, state_batch):
        state_batch = self.pca.transform(state_batch)

        # FIX THIS
        state_batch = self.relu1(self.linear1(state_batch))
        state_batch = self.relu2(self.linear2(state_batch))
        state_batch = self.relu3(self.linear3(state_batch))
        result = self.out_layer(state_batch)
        return result

    def train_model(self, memory, target_network=None):
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
            next_state_values[non_final_mask] = self.target_network[0].forward(non_final_next_states).max(1)[0]
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
        if self.anneal:
            self.scheduler.step()
        self.train_counter += 1

        if self.use_target_network and self.train_counter % self.target_update == 0:
            self.sync_target_network()

        return loss.data[0] / len(state_action_values)

    def compute_sample_Q(self, sample_states):
        res = self.forward(sample_states).data.max(1)[0].mean()
        return float(res)

    def create_target_network(self):
        self.target_network = [DQNGS(env = self.env, batch_sz = self.batch_size,
                                    lr = self.learning_rate, gamma = self.gamma,
                                    regularization = self.regularization,
                                    target_update = 0,
                                    anneal = self.anneal, loss = self.loss_name)]
        self.target_network[0].load_state_dict(self.state_dict())
        self.target_network[0].eval() # can't train target_network again

    def cuda(self):
        super(DQNGS, self).cuda()
        print(self.use_target_network)
        if self.use_target_network:
            self.target_network[0].cuda()

    def load_state_dict(self, state_dict):        
        if self.use_target_network:
            super(DQNGS, self).load_state_dict(state_dict)
            self.target_network[0].load_state_dict(state_dict)

    def sync_target_network(self):
        self.target_network[0].load_state_dict(self.state_dict())
