'''
This script brings all the parts together and should
serve as a script that represents the current functionality
of the project
'''
from internal.music2vec import ScoreFetcher, ScoreToWord, ScoreToVec
from internal.instantiate import *
from internal.type_models import BayesianGaussianTypeModel

import os.path
import pickle

import matplotlib.pyplot as plt
import numpy as np

from hmmlearn import hmm
from music21 import converter

SCORE_PATH = r'data/score_cache.json'
SCORE_WORD_PATH = r'data/score_word_cache.json'
EMBEDDING_PATH = r'data/embedding.wv'
TYPE_MODEL_PATH = r'data/type_model.pickle'
HMM_PATH = r'data/hmm.pickle'

# Prepare for saving
no_object_computes = True
def should_compute(obj : object, prompt: str = 'Load cache'):
    global no_object_computes
    if no_object_computes:
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
if should_compute(myScoreToWord, 'Load cached words'):
    # Only load the score fetcher if you are computing words
    myScoreFetcher = ScoreFetcher(SCORE_PATH)
    if should_compute(myScoreFetcher, 'Load cached scores'):
        myScoreFetcher.fetch()

    myScoreToWord.process(myScoreFetcher.scores)

print('Training embedding model...')
score_word_to_vec = ScoreToVec(myScoreToWord.scores, path=EMBEDDING_PATH)

print('Training type model...')
myTypeModel = BayesianGaussianTypeModel(path=TYPE_MODEL_PATH)
if not os.path.exists(TYPE_MODEL_PATH):
    myTypeModel.fit(score_word_to_vec.embedding.vectors)
    myTypeModel.save_model(TYPE_MODEL_PATH)

labels = myTypeModel.predict(score_word_to_vec.embedding.vectors)
print(labels)

plt.hist(labels, bins=32)
plt.show()


#################
# Now Train HMM #
#################
# TODO: formalize into object
print('Training generative model...')
if not os.path.exists(HMM_PATH):
    word_to_label = {}

    vocab = score_word_to_vec.vocab()
    for word in vocab:
        idx = vocab[word].index
        word_to_label[word] = labels[idx]

    def _text_to_seq(text):
        return np.array([[word_to_label[word]] for word in text])

    sequences = [_text_to_seq(text) for text in myScoreToWord.scores]

    # Now actually train
    hmm_model = hmm.MultinomialHMM(n_components=16)
    lengths = [len(seq) for seq in sequences]
    sequences = np.concatenate(sequences)
    hmm_model.fit(sequences, lengths=lengths)
    print(hmm_model.transmat_)
    with open(HMM_PATH, "wb") as file: pickle.dump(hmm_model, file)
else:
    hmm_model = None
    with open(HMM_PATH, "rb") as file: hmm_model = pickle.load(file)

###################
# Test Generation #
###################
print('Testing generation...')

types, Z = hmm_model.sample(40)
types = types.flatten()
print(types)

token = to_token(types, myTypeModel, score_word_to_vec)
score = to_score(token, chord_with_ties_texture, duration=1)
score.show('musicxml')
