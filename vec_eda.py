'''
This script brings all the parts together and should
serve as a script that represents the current functionality
of the project
'''
from internal.music2vec import ScoreFetcher, ScoreToWord, ScoreToVec
from internal.type_models import BayesianGaussianTypeModel

import os.path
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.manifold import TSNE

from hmmlearn import hmm
from music21 import converter

SCORE_PATH = r'data/score_cache.json'
SCORE_WORD_PATH = r'data/score_word_cache.json'
EMBEDDING_PATH = r'data/embedding.wv'
TYPE_MODEL_PATH = r'data/type_model.pickle'
HMM_PATH = r'data/hmm.pickle'

# Prepare for saving
no_object_computes = True
def should_compute(obj : object, prompt: str = 'Load cache', force_ask: bool = False):
    global no_object_computes
    if no_object_computes or force_ask:
        inp = input(prompt + ' [y/n]?')
        if len(inp) > 0 and inp[0] == 'n':
            no_object_computes = False
            return True
        obj.load_cache()
        return False
    else:
        return True

print('Loading/processing scores...')
myScoreToWord = ScoreToWord(SCORE_WORD_PATH)
myScoreToWord.load_cache()

print('Training embedding model...')
score_word_to_vec = ScoreToVec(myScoreToWord.scores, path=EMBEDDING_PATH, size=3)
vecs = score_word_to_vec.embedding.vectors

print('orig_dim', len(vecs[0]))

X_emb = TSNE(n_components=3).fit_transform(vecs)

# Now generate plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([i[0] for i in X_emb],
        [i[1] for i in X_emb],
        [i[2] for i in X_emb])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
