'''
This script brings all the parts together and should
serve as a script that represents the current functionality
of the project
'''
from internal.music2vec import ScoreFetcher, ScoreToWord, ScoreToVec
from internal.instantiate import *
from internal.type_models import BayesianGaussianTypeModel
from internal.neural import GRUTypeNet

import os
import os.path
import sys
import pickle
import datetime
import json

import matplotlib.pyplot as plt
import numpy as np

from hmmlearn import hmm
from music21 import converter

# choices: 'gru' or 'hmm'
TYPE_GENERATOR = 'hmm'
if len(sys.argv) > 1:
    TYPE_GENERATOR = sys.argv[1]

SCORE_PATH = r'data/score_cache.json'
SCORE_WORD_PATH = r'data/score_word_cache.json'
EMBEDDING_PATH = r'data/embedding.wv'
TYPE_MODEL_PATH = r'data/type_model.pickle'
HMM_PATH = r'data/hmm.pickle'
GRU_PATH = r'data/type_gru_model.pt'
OUTPUT_PATH = r'output/'

# Enables or disables the conditional generation of
# terms by previous
DO_CONDITIONAL_GENERATION = True
SMOOTH_PARAM = 0.01

# Prepare for saving
def should_compute(obj : object, prompt: str = 'Load cache'):
    inp = input(prompt + ' [y/n]?')
    if len(inp) > 0 and inp[0] == 'n':
        no_object_computes = False
        return True
    obj.load_cache()
    return False


print('Loading/processing scores...')
myScoreToWord = ScoreToWord(SCORE_WORD_PATH)
if should_compute(myScoreToWord, 'Load cached words'):
    # Only load the score fetcher if you are computing words
    myScoreFetcher = ScoreFetcher(SCORE_PATH)
    if should_compute(myScoreFetcher, 'Load cached scores'):
        myScoreFetcher.fetch()

    myScoreToWord.process(myScoreFetcher.scores, test_split=0.05)

print('Training embedding model...')
score_word_to_vec = ScoreToVec(myScoreToWord.scores, path=EMBEDDING_PATH)

print('Training type model...')
myTypeModel = BayesianGaussianTypeModel(path=TYPE_MODEL_PATH, embedding=score_word_to_vec.embedding,
                  do_conditional=DO_CONDITIONAL_GENERATION, smooth=SMOOTH_PARAM)
if not os.path.exists(TYPE_MODEL_PATH):
    myTypeModel.fit(score_word_to_vec.embedding.vectors, myScoreToWord.scores)
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
NUM_TYPES = 32
if TYPE_GENERATOR == 'hmm':
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
        type_gen_model = hmm.MultinomialHMM(n_components=16)
        lengths = [len(seq) for seq in sequences]
        sequences = np.concatenate(sequences)
        type_gen_model.fit(sequences, lengths=lengths)
        print(type_gen_model.transmat_)
        with open(HMM_PATH, "wb") as file: pickle.dump(type_gen_model, file)
    else:
        type_gen_model = None
        with open(HMM_PATH, "rb") as file: type_gen_model = pickle.load(file)
elif TYPE_GENERATOR == 'gru':
    if not os.path.exists(GRU_PATH):
        word_to_label = {}
        vocab = score_word_to_vec.vocab()
        for word in vocab:
            idx = vocab[word].index
            word_to_label[word] = labels[idx]

        def _text_to_seq(text):
            return np.array([word_to_label[word] for word in text])

        sequences = [_text_to_seq(text) for text in myScoreToWord.scores]

        # Now actually train
        type_gen_model = GRUTypeNet(input_dim=NUM_TYPES, hidden_dim=16, output_dim=NUM_TYPES)
        type_gen_model.fit(sequences, EPOCHS=40)
        with open(GRU_PATH, "wb") as file: pickle.dump(type_gen_model, file)
    else:
        type_gen_model = None
        with open(GRU_PATH, 'rb') as file: type_gen_model = pickle.load(file)
else:
    print('!!!!WARN!!!! SUPPLIED TYPE GENERATOR ({}) IS NOT RECOGNIZED'.format(TYPE_GENERATOR))

###################
# Test Generation #
###################
print('Testing generation...')

types, Z = type_gen_model.sample(40)
types = types.flatten()
print(types)

token = to_token(types, myTypeModel, score_word_to_vec)

# save output
if not os.path.isdir(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
song_format = ['_'.join(i) for i in token] # put the notes together in one string
open(OUTPUT_PATH + 'song_main_{}.json'.format(datetime.datetime.now()), 'w').write(json.dumps(song_format))

# now play actual song
score = to_score(token, chord_with_ties_texture, duration=1)
score.show('musicxml')
