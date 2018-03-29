import gym

class CartPoleGame():

    def __init__(self):
        self.file_prefix = 'CartPole_'
        self.env = gym.make('CartPole-v0').unwrapped
        self.screen_width = 600

    def modify_screen(self, screen):
        return screen