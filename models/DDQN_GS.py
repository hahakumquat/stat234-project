import numpy as np

from DQN_GS import DQNGS

class DDQNGS():

    def __init__(self, env, batch_sz=128, lr=0.1, gamma=0.99):
        self.env = env
        self.batch_size = batch_sz
        self.lr = lr
        self.gamma = gamma
        self.modelA = DQNGS(env=self.env, batch_sz=self.batch_size, lr=self.lr, gamma=self.gamma)
        self.modelB = DQNGS(env=self.env, batch_sz=self.batch_size, lr=self.lr, gamma=self.gamma)

    def forward(self, state_batch):
        result = (self.modelA.forward(state_batch) + self.modelB.forward(state_batch)) / 2
        return result

    def train_model(self, memory, target_network):
        if np.random.random() < 0.5:
            return self.modelA.train_model(memory, self.modelB)
        else:
            return self.modelB.train_model(memory, self.modelA)
        
    def compute_sample_Q(self, sample_states):
        return (self.modelA.compute_sample_Q(sample_states) + self.modelA.compute_sample_Q(sample_states)) / 2
