'''
This script brings all the parts together and should
serve as a script that represents the current functionality
of the project
'''
# Local imports
from music2vec import ScoreToWord, ScoreToVec
from type_models import BayesianGaussianTypeModel

# General imports
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from hmmlearn import hmm

#####################################################
# Get scores and corresponding word representations
#####################################################
score_word_cache = '.score_word_cache.json'
myScoreToWord = ScoreToWord()
q = input('Load cached words? [y/n]:')
# Get score_words variable filled with scores converted to words
if len(q) > 0 and q[0] == 'y':
    score_words = myScoreToWord.load_score_words(score_word_cache)
else:
    print('Loading/processing scores...')
    scores = myScoreToWord.get_scores()
    score_words = list(tqdm(myScoreToWord.scores_to_text(scores)))
    myScoreToWord.save_score_words(score_words, score_word_cache)

##############################
# Now train a Word2Vec model #
##############################
EMBEDDING_PATH = r'embedding.wv'
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
myTypeModel.fit(vec_model.vectors)
labels = myTypeModel.predict_multi(vec_model.vectors)

plt.hist(labels, bins=32)


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
X, Z = hmm_model.sample(20)
print(X)
