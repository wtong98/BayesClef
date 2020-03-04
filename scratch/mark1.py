"""
Scratch file to play around with basic concepts. Formatted as Hydrogen
notebook.

author: William Tong (wlt2115)
date: 2-23-2020
"""

# <codecell>
from collections import defaultdict

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from music21 import corpus
from music21 import interval
from music21 import note

from pomegranate import HiddenMarkovModel, NormalDistribution

from sklearn.mixture import BayesianGaussianMixture

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

EMBEDDING_PATH = r'embedding.wv'

# <codecell>
bach_bundle = corpus.search('bach', 'composer')
scores = [metadata.parse() for metadata in bach_bundle]

# <codecell>
# TODO: split sentences by measure?
def convert_to_texts(scores, sampling_rate=0.5):
    for score in scores:
        normalized_score = _transpose_to_c(score)
        text = _to_text(normalized_score, sampling_rate)
        yield text

def _transpose_to_c(score) -> 'Score':
    ky = score.analyze('key')
    home = note.Note(ky.tonicPitchNameWithCase)
    target = note.Note('c')
    int = interval.Interval(home, target)

    return score.transpose(int)


def _to_text(score, sampling_rate) -> list:
    notes = score.flat.getElementsByClass(note.Note)
    hist = _bin(notes, sampling_rate)
    end = score.flat.highestOffset

    text = [_to_word(hist[i]) for i in np.arange(0, end, sampling_rate)]
    return text


def _bin(notes, sampling_rate) -> defaultdict:
    hist = defaultdict(list)

    for note in notes:
        offset = note.offset
        halt = offset + note.duration.quarterLength

        if _precise_round(offset % sampling_rate) != 0:
            offset = _precise_round(offset - (offset % sampling_rate))
        if _precise_round(halt % sampling_rate) != 0:
            halt = _precise_round(halt + (sampling_rate - halt % sampling_rate))

        while offset < halt:
            hist[offset].append(note)
            offset += sampling_rate

    return hist


def _to_word(notes, rest_word="XREST") -> str:
    if len(notes) == 0:
        return rest_word

    ordered_notes = sorted(notes, key=lambda n: n.pitch.midi, reverse=True)
    word = ''.join([note.name for note in ordered_notes])
    return word


def _precise_round(val, precision=10):
    return round(val * precision) / precision


# <codecell>
# TODO: tune word2vec params
# TODO: evaluate freqeuency histogram
texts = list(tqdm(convert_to_texts(scores)))
# model = Word2Vec(sentences=texts,
#                  size=32,
#                  min_count=1,
#                  window=4,
#                  workers=4,
#                  sg=1)
#
# model.wv.save(EMBEDDING_PATH)


# <codecell>
wv = KeyedVectors.load(EMBEDDING_PATH)
# TODO: needs tuning, has trouble converging
mixture = BayesianGaussianMixture(n_components=32)
mixture.fit(wv.vectors)

labels = mixture.predict(wv.vectors)
plt.hist(labels, bins=32)

# <codecell>
word_to_label = {}

for word in wv.vocab:
    idx = wv.vocab[word].index
    word_to_label[word] = labels[idx]

def _text_to_seq(text):
    return np.array([word_to_label[word] for word in text])

sequences = [_text_to_seq(text) for text in texts]

# <codecell>
hmm = HiddenMarkovModel.from_samples(NormalDistribution, # TODO: identify discrete distribution
                                     n_components=16,
                                     X=sequences)

# <codecell>
hmm.dense_transition_matrix()




















# <codecell>
