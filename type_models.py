'''
Provides various classes that allow the unsupervised extraction
of types from the score data
'''
from sklearn.mixture import BayesianGaussianMixture

class TypeModel:
    def fit(self, vectors):
        pass

    def predict_multi(self, vectors):
        return [self.predict(i) for i in vectors]

    def predict(self, vector):
        return 0

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
