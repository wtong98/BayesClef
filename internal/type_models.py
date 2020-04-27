'''
Provides various classes that allow the unsupervised extraction
of types from the score data
'''
import os.path
import pickle
import random

from collections import defaultdict, Counter

import numpy as np
from sklearn.mixture import BayesianGaussianMixture
import scipy.stats

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
    def __init__(self, embedding, n_components=32, path: str = '', load=True, smooth: int = 0.01, do_conditional: bool = True):
        ''' embedding = Word2Vec style embedding
        '''
        self.embedding = embedding
        self.n_grams = [2, 3] # set up for 2 and 3-gram combo
        self.do_conditional = do_conditional
        self.smooth = smooth
        self.n = n_components
        if os.path.exists(path) and load:
            self.load_model(path)

        # Set up function caching
        self.norm_prob_cache = {}

    def fit(self, vectors, scores):
        # Fit Bayesian Gaussian Mixture model
        self.train_vectors = vectors
        self.mixture = BayesianGaussianMixture(n_components=self.n)
        self.mixture.fit(vectors)
        # Fit conditional dependence model
        # Compute n-gram model
        gram_map = { 2: defaultdict(Counter), 3: defaultdict(Counter) }
        for score in scores:
            for n_gram in self.n_grams:
                for i in range(len(score)-n_gram+1):
                    followed = score[i+n_gram-1]
                    gram = tuple(score[i:i+n_gram-1])
                    gram_map[n_gram][gram][followed] += 1
        self.gram_map = gram_map
        

    def predict(self, vector):
        return self.mixture.predict(vector)

    def conditional_prob(self, option, prev_words):
        ''' Produces a smoothed n-gram probability
            but not divided by the total sum and thus
            not strictly a PDF but rather a constant
            scaling of one
        '''
        smooth_count = 0
        for n_gram in self.n_grams:
            prev_key = tuple(prev_words[-(n_gram-1):])
            followers = self.gram_map[n_gram][prev_key]
            smooth_count += followers[option] + self.smooth #*((n_gram-1)**2) + self.smooth
        # TODO: Maybe make into a true pmf (i.e. sum to 1) but 
        # this won't improve results or change anything
        return smooth_count

    def emit(self, type_id, prev_words=[]):
        mean = self.mixture.means_[type_id,:]
        cov = self.mixture.covariances_[type_id,:,:]
        if len(prev_words) == [] or not self.do_conditional: # if not conditioning on previous
            draw = np.random.multivariate_normal(mean, cov)
        else:
            opts = list(self.embedding.vocab)
            weights = []
            if not type_id in self.norm_prob_cache:
                self.norm_prob_cache[type_id] = [scipy.stats.multivariate_normal(mean, cov).pdf(self.embedding[wrd]) for \
                              wrd in opts]
            for i, wrd in enumerate(opts):
                norm_prob = self.norm_prob_cache[type_id][i]
                ngram_prob = self.conditional_prob(wrd, prev_words)
                weights.append(norm_prob*ngram_prob)
            draw = random.choices(population=opts, k=1, weights=weights)[0]
            print(draw, type_id)
            # convert to vector for legacy support
            draw = self.embedding[draw]
        return draw

    def save_model(self, file_name):
        s = pickle.dumps({ 'mixture': self.mixture, 'gram_map': self.gram_map })
        open(file_name, 'wb').write(s)

    def load_model(self, file_name):
        s = open(file_name, 'rb').read()
        data = pickle.loads(s)
        self.mixture = data['mixture']
        self.gram_map = data['gram_map']
