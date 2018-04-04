import numpy as np

from DQN_GS import DQNGS

class DDQNGS():
    def __init__(self, env, batch_sz=128, lr=0.1, gamma=0.99, regularization=0.0001, target_update=0, anneal=False, loss="Huber"):

        self.env = env
        self.batch_size = batch_sz
        self.learning_rate = lr
        self.gamma = gamma
        self.regularization = regularization
        self.modelA = DQNGS(env=self.env, batch_sz=self.batch_size, 
                            lr=self.learning_rate, gamma=self.gamma, 
                            regularization=0.0001, target_update=0,
                            anneal=anneal, loss=loss)
        self.modelB = DQNGS(env=self.env, batch_sz=self.batch_size, 
                            lr=self.learning_rate, gamma=self.gamma, 
                            regularization=0.0001, target_update=0,
                            anneal=anneal, loss=loss)

        # figure out my parameters
        self.lr_annealer = self.modelA.lr_annealer
        self.optim_name = self.modelA.optim_name
        self.loss_name = self.modelA.loss_name

        self.train_counter = 0

        self.use_target_network = 'sort of. this is a DDQN'

    def forward(self, state_batch):
        result = (self.modelA.forward(state_batch) + self.modelB.forward(state_batch)) / 2
        return result

    def train_model(self, memory):
        self.train_counter += 1
        if np.random.random() < 0.5:
            return self.modelA.train_model(memory, self.modelB)
        else:
            return self.modelB.train_model(memory, self.modelA)
        
    def compute_sample_Q(self, sample_states):
        return (self.modelA.compute_sample_Q(sample_states) + self.modelA.compute_sample_Q(sample_states)) / 2

    def cuda():
        self.modelA.cuda()
        self.modelB.cuda()
