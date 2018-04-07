import numpy as np
import os
import pickle
from sklearn.decomposition import PCA as sklPCA
from sklearn.preprocessing import StandardScaler
import sys

# stat234-project
root = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(root)
root = os.path.dirname(os.getcwd())
os.chdir(root)

class PCA():

    def __init__(self, states_file, original_dim=(80, 80), n=100, batch_size=100):
        self.original_dim = original_dim

        self.n_pixels = original_dim[0] * original_dim[1]

        states = np.genfromtxt(states_file, delimiter=',')
        states /= 255 # states_file is made up of ints

        self.std_scale = StandardScaler().fit(states)
        states_std = self.std_scale.transform(states)

        self.pca = sklPCA(n_components=n)
        self.pca.fit(states_std)

    def transform(self, state):
        y = state.reshape(len(state), -1)
        y_std = self.std_scale.transform(y)
        return self.pca.transform(y_std)
        
    def inv_transform(self, transformed):
        return self.std_scale.inverse_transform(
            self.pca.inverse_transform(transformed)).reshape(
            (len(transformed), self.original_dim[0], self.original_dim[1]))

if __name__ == '__main__':
    for game in ['CartPole', 'Acrobot', 'MountainCar']:
        pca = PCA('data/states/' + game + '_states.csv', n=0.99)
        with open('data/states/' + game + '_PCA.pkl', 'wb') as f:
            pickle.dump(pca, f)
