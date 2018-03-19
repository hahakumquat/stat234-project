import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

from Logger import Logger
from ReplayMemory import Transition

class DQN(nn.Module):

    def __init__(self, env, batch_sz=128, lr=0.1, gamma=0.999, loss_filename='dqn_cartpole_losses.pdf'):
        super(DQN, self).__init__()

        ## DQN architecture
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.LeakyReLU(0.05)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.LeakyReLU(0.05)
        
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.LeakyReLU(0.05)
        
        self.out_layer = nn.Linear(576, 2)

        self.env = env
        self.batch_size = batch_sz
        self.learning_rate = lr
        self.gamma = gamma
        
        self.optimizer = optim.RMSprop(self.parameters(),
                                       lr=self.learning_rate)
        self.loss_function = F.smooth_l1_loss
        self.losses = []
        self.loss_filename = loss_filename

    def forward(self, state_batch):
        state_batch = self.relu1(self.bn1(self.conv1(state_batch)))
        state_batch = self.relu2(self.bn2(self.conv2(state_batch)))
        state_batch = self.relu3(self.bn3(self.conv3(state_batch)))
        state_batch = state_batch.view(state_batch.shape[0], -1)
        return self.out_layer(state_batch)

    def train(self, state_batch, action_batch, reward_batch, next_state_values):
        transitions = memory.sample(self.batch_size)
        # stackoverflow: 
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
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
        next_state_values = Variable(torch.zeros(len(state_batch)).type(torch.FloatTensor))
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
        
        if len(self.losses) % 100 == 0:
            self.plot_losses()

    def plot_losses(self):
        plt.plot(self.losses)
        plt.title("Per-SARS Huber Loss")
        plt.savefig(self.loss_filename)
        plt.close()
