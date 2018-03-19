import gym

class AcrobotGame():

    def __init__(self):
        self.file_prefix = 'acrobot_'
        self.env = gym.make('Acrobot-v1').unwrapped

    def modify_screen(self, screen):
        return screen
    
