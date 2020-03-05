'''
Provides various classes that allow the unsupervised extraction
of types from the score data
'''
import os.path
import pickle

import numpy as np
from sklearn.mixture import BayesianGaussianMixture

# TODO: close open file handles

class TypeModel:
    def fit(self, vectors):
        pass

    def predict(self, vector):
        return 0

    def emit(self, type_id):
        return None

    def save_model(self, file_name):
        pass

    def load_model(self, file_name):
        pass

class BayesianGaussianTypeModel(TypeModel):
    def __init__(self, n_components=32, path: str = '', load=True):
        self.n = n_components
        if os.path.exists(path) and load:
            self.load_model(path)

    def fit(self, vectors):
        self.train_vectors = vectors
        self.mixture = BayesianGaussianMixture(n_components=self.n)
        self.mixture.fit(vectors)

    def predict(self, vector):
        return self.mixture.predict(vector)

    def emit(self, type_id):
        mean = self.mixture.means_[type_id,:]
        cov = self.mixture.covariances_[type_id,:,:]
        draw = np.random.multivariate_normal(mean, cov)
        return draw

    def save_model(self, file_name):
        s = pickle.dumps(self.mixture)
        open(file_name, 'wb').write(s)

    def load_model(self, file_name):
        s = open(file_name, 'rb').read()
        self.mixture = pickle.loads(s)
