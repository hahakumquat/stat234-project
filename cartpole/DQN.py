import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

from PIL import Image

class DQN(nn.Module):

    def __init__(self, env, screen_width=600, batch_sz=128, lr=0.1, gamma=0.999):
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
        
        self.out_layer = nn.Linear(448, 2)

        self.env = env
        self.screen_width = 600
        self.batch_size = batch_sz
        self.learning_rate = lr
        self.gamma = gamma
        
        self.optimizer = optim.RMSprop(self.parameters(),
                                       lr=self.learning_rate)
        self.loss_function = nn.SmoothL1Loss

    def forward(self, state_batch):
        state_batch = self.truncate(state_batch)
        state_batch = self.relu1(self.bn1(self.conv1(state_batch)))
        state_batch = self.relu2(self.bn2(self.conv2(state_batch)))
        state_batch = self.relu3(self.bn3(self.conv3(state_batch)))
        return self.out_layer(state_batch.view(state_batch.size(0), -1))

    def train(self, state_batch, action_batch, reward_batch, next_state_values):

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.forward(state_batch).gather(1, action_batch)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = self.loss_function(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # state_action_values = self.forward(states)
        # loss = self.lossFunction(state_action_values, expected_state_action_values)
        # self.optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # print(loss.data)

    def truncate(self, state_batch):
        return [self.cut_screen(screen) for screen in state_batch]

    # helper functions
    def get_cart_location(self):
        world_width = self.env.x_threshold * 2
        scale = self.screen_width / world_width
        return int(env.state[0] * scale + self.screen_width / 2.0)  # MIDDLE OF CART

    def resize(self, screen):
        rsz = T.Compose([T.ToPILImage(),
                T.Scale(40, interpolation=Image.CUBIC),
                T.ToTensor()])
        return rsz(screen)
        
    def cut_screen(self, screen):
        print(screen.shape)
        screen = screen.transpose((2, 0, 1))  # transpose into torch order (CHW)
        # Strip off the top and bottom of the screen
        screen = screen[:, 160:320]
        view_width = 320 # can be like 60
        # cart_location = get_cart_location()
        # if cart_location < view_width // 2:
        #     slice_range = slice(view_width)
        # elif cart_location > (self.screen_width - view_width // 2):
        #     slice_range = slice(-view_width, None)
        # else:
        #     slice_range = slice(cart_location - view_width // 2,
        #                         cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        # screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.array(screen) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        # output 1x3x40x80
        return resize(screen).unsqueeze(0).type(Tensor)


