import gym

class MountainCarGame():

    def __init__(self):
        self.file_prefix = 'MountainCar_'
        self.env = gym.make('MountainCar-v0').unwrapped

    def modify_screen(self, screen):
        return screen
    
