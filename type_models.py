'''
Provides various classes that allow the unsupervised extraction
of types from the score data
'''
from sklearn.mixture import BayesianGaussianMixture
import pickle

class TypeModel:
    def fit(self, vectors):
        pass

    def predict_multi(self, vectors):
        return [self.predict(i) for i in vectors]

    def predict(self, vector):
        return 0

    def emit(self, type_id):
        return None

    def save_model(self, file_name):
        pass

    def load_model(self, file_name):
        pass

class BayesianGaussianTypeModel(TypeModel):
    def __init__(self):
        pass

    def fit(self, vectors):
        self.train_vectors = vectors
        self.mixture = BayesianGaussianMixture(n_components=32)
        self.mixture.fit(vectors)

    def predict(self, vector):
        return self.mixture.predict([vector])[0]

    def predict_multi(self, vectors):
        return self.mixture.predict(vectors)

    def emit(self, type_id):
        # TODO: Directly sample from Gaussian
        # This is a super hacky way to use the built in sample() method
        max_iter = 100
        curr_iter = 0
        while curr_iter < max_iter:
            curr_iter += 1
            X, y = self.mixture.sample()
            if y[0] == type_id:
                return X[0]
        print('Failed to get sample from type', type_id)
        return None

    def save_model(self, file_name):
        s = pickle.dumps(self.mixture)
        open(file_name, 'wb').write(s)

    def load_model(self, file_name):
        s = open(file_name, 'rb').read()
        self.mixture = pickle.loads(s)
