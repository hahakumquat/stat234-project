import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from Logger import Logger

class DQNGS(nn.Module):

    def __init__(self, env, batch_sz=128, lr=0.1, gamma=0.999, loss_filename='dqn_gs_cartpole_losses.pdf'):
        super(DQNGS, self).__init__()

        ## DQN architecture
        self.conv1 = nn.Conv2d(1, 8, kernel_size=8, stride=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.LeakyReLU(0.01)
        
        self.conv2 = nn.Conv2d(8, 8, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.LeakyReLU(0.01)

        self.conv3 = nn.Conv2d(8, 16, kernel_size=4, stride=1)
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
        self.loss_filename = loss_filename
        self.loss_log = Logger('losses.csv')

    def forward(self, state_batch):
        state_batch = self.relu1(self.bn1(self.conv1(state_batch)))
        state_batch = self.mp(self.relu2(self.bn2(self.conv2(state_batch))))
        state_batch = self.mp(self.relu3(self.bn3(self.conv3(state_batch))))
        state_batch = state_batch.view(state_batch.shape[0], -1)
        return self.out_layer(state_batch)

    def train(self, state_batch, action_batch, reward_batch, next_state_values):

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.forward(state_batch)
        state_action_values = state_action_values.gather(1, action_batch)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        self.losses.append(loss.data / len(state_action_values))
        self.loss_log.log(loss.data / len(state_action_values))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
    #     if len(self.losses) % 100 == 0:
    #         self.plot_losses()

    # def plot_losses(self):
    #     plt.plot(self.losses)
    #     plt.title("Per-SARS Huber Loss")
    #     plt.savefig(self.loss_filename)
    #     plt.close()
