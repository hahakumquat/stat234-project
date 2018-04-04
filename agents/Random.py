class Random():

    def __init__(self, model, env):
        self.env = env
        self.steps_done = 0

    def select_action(self, state):
        return self.env.action_space.sample()
