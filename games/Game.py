import gym

class Game():

    def __init__(self, game_name):
        self.env = gym.make(game_name).unwrapped
        self.file_prefix = game_name.split('-')[0] + '_'

    def modify_screen(self, screen):
        return screen
    
