import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.LeakyReLU(0.05)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.LeakyReLU(0.05)
        
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.LeakyReLU(0.05)
        
        self.out_layer = nn.Linear(448, 2)

        self.optimizer = optim.RMSprop(self.parameters(),
                                       lr=self.learning_rate)
        self.lossFunction = nn.SmoothL1_Loss
        
        self.batch_size = 100
        self.learning_rate = 0.1        

    def forward(self, states):
        states = self.relu1(self.bn1(self.conv1(states)))
        states = self.relu2(self.bn2(self.conv2(states)))
        states = self.relu3(self.bn3(self.conv3(states)))
        return self.out_layer(states.view(states.size(0), -1))

    def train(self, states):
        state_action_values = self.forward(states)
        loss = self.lossFunction(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.data)
