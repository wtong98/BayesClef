'''
This script brings all the parts together and should
serve as a script that represents the current functionality
of the project
'''
# Local imports
from internal.music2vec import ScoreToWord, ScoreToVec
from internal.type_models import BayesianGaussianTypeModel

# General imports
import matplotlib.pyplot as plt
import numpy as np

from hmmlearn import hmm
from music21 import converter
from tqdm import tqdm

SCORE_WORD_PATH = '.score_word_cache.json'
EMBEDDING_PATH = r'.embedding.wv'
TYPE_MODEL_PATH = '.type_model.pickle'

#####################################################
# Get scores and corresponding word representations
#####################################################
myScoreToWord = ScoreToWord()
q = input('Load cached words? [y/n]:')
# Get score_words variable filled with scores converted to words
if len(q) > 0 and q[0] == 'y':
    score_words = myScoreToWord.load_score_words(SCORE_WORD_PATH)
else:
    print('Loading/processing scores...')
    scores = myScoreToWord.get_scores()
    score_words = list(tqdm(myScoreToWord.scores_to_text(scores)))
    myScoreToWord.save_score_words(score_words, score_word_cache)

##############################
# Now train a Word2Vec model #
##############################
myScoreToVec = ScoreToVec(EMBEDDING_PATH)
q = input('Load cached embeddings? [y/n]:')
if len(q) > 0 and q[0] == 'y':
    vec_model = myScoreToVec.load_model()
else:
    print('Training Word2Vec model...')
    vec_model = myScoreToVec.train_model(score_words)

print(myScoreToVec.decode(vec_model.vectors[0]))


####################################
# Now Train Gaussian Mixture Model #
####################################
print('Training type model...')
myTypeModel = BayesianGaussianTypeModel()
q = input('Load cached type model? [y/n]:')
if len(q) > 0 and q[0] == 'y':
    vec_model = myScoreToVec.load_model()
    myTypeModel.load_model(TYPE_MODEL_PATH)
else:
    myTypeModel.fit(vec_model.vectors)
    myTypeModel.save_model(TYPE_MODEL_PATH)
labels = myTypeModel.predict_multi(vec_model.vectors)
print(labels)

plt.hist(labels, bins=32)
plt.show()


#################
# Now Train HMM #
#################
print('Training generative model...')
word_to_label = {}

vocab = myScoreToVec.vocab()
for word in vocab:
    idx = vocab[word].index
    word_to_label[word] = labels[idx]

def _text_to_seq(text):
    return np.array([[word_to_label[word]] for word in text])

sequences = [_text_to_seq(text) for text in score_words]

# Now actually train
hmm_model = hmm.GaussianHMM(n_components=16)
lengths = [len(seq) for seq in sequences]
sequences = np.concatenate(sequences)
hmm_model.fit(sequences, lengths=lengths)
print(hmm_model.transmat_)

###################
# Test Generation #
###################
print('Testing generation...')
# Generate types
X, Z = hmm_model.sample(40)

# Discretize types
types = [int(round(i[0], 0)) for i in X]
print(types)

# Get randomized vector from type distribution
vec_out = []
for i in types:
    vec_out.append(myTypeModel.emit(i))

# convert into sequence of note combinations
new_score = []
for vec in vec_out:
    try:
        if vec == None: # handle if no emition could be generated
            new_score.append('XREST')
    except ValueError: # thrown if vec is a vector and not None
        new_score.append(myScoreToVec.decode(vec))
print(new_score)

# Make it ingestable by music21 and display music
music_representation = 'tinynotation: 4/4 '
new_note_sequences = []
for i in new_score:
    notes = i.split('_')
    print(notes)
    num = len(notes)
    if num == 1 and notes[0] == 'XREST':
        continue # TODO: Make into actual rest
    # TODO: Music21 only supports multiples of 4 or something so 5 results in errors
    cap = 4
    if num > cap:
        num = cap
    start_note = notes[0] + str(4*num)
    new_notes = ' '.join([start_note] + notes[1:num])
    new_note_sequences += [new_notes]
music_representation += ' '.join(new_note_sequences)
print(music_representation)
# Now play new music
converter.parse(music_representation).show()
