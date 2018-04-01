import numpy as np
from sklearn.decomposition import IncrementalPCA as sklPCA

class PCA():

    def __init__(self, states, n = 100):
        self.pca = sklPCA(n_components = n)
        X = states.reshape((len(states), -1))
        self.original_dim = states[0].shape
        self.pca.fit(X)

    def transform(self, state):
        # 10 x 1
        y = state.reshape(len(state),-1)
        return self.pca.transform(y)
        
    def inv_transform(self, transformed):
        return self.pca.inverse_transform(transformed).reshape((len(transformed), self.original_dim[0], self.original_dim[1]))

