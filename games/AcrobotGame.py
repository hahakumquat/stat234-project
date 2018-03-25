import gym

class AcrobotGame():

    def __init__(self):
        self.file_prefix = 'Acrobot_'
        self.env = gym.make('Acrobot-v1').unwrapped

    def modify_screen(self, screen):
        return screen
    
