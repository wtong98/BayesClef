import json
import os
import os.path
from collections import defaultdict

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from music21 import corpus
from music21 import interval
from music21 import note
from tqdm import tqdm

import numpy as np


class ScoreToWord:
    '''
    Loads scores and converts them into word form
    '''

    @staticmethod
    def query_scores(artist='bach', debug=False):
        bundle = corpus.search(artist, 'composer')
        if debug:
            print('Loading {0} scores'.format(len(bundle)))
        return [metadata.parse() for metadata in bundle]


    def __init__(self, raw_scores: list = [], path: str = '', load=True):
        if os.path.exists(path) and load:
            self.scores = self._load_score_words(path)
        else:
            self.scores = list(tqdm(self.scores_to_text(raw_scores)))
            self._save_score_words(self.scores, path)

    def scores_to_text(self, scores, sampling_rate=0.5):
        for score in scores:
            yield self.score_to_text(score, sampling_rate)

    def score_to_text(self, score, sampling_rate=0.5):
        normalized_score = self._transpose_to_c(score)
        return self._to_text(normalized_score, sampling_rate)

    def _transpose_to_c(self, score) -> 'Score':
        ky = score.analyze('key')
        home = note.Note(ky.tonicPitchNameWithCase)
        target = note.Note('c')
        int = interval.Interval(home, target)
        return score.transpose(int)

    def _to_text(self, score, sampling_rate) -> list:
        notes = score.flat.getElementsByClass(note.Note)
        hist = self._bin(notes, sampling_rate)
        end = score.flat.highestOffset

        text = [self._to_word(hist[i]) for i in np.arange(0, end, sampling_rate)]
        return text

    def _bin(self, notes, sampling_rate) -> defaultdict:
        hist = defaultdict(list)

        for note in notes:
            offset = note.offset
            halt = offset + note.duration.quarterLength

            if self._precise_round(offset % sampling_rate) != 0:
                offset = self._precise_round(offset - (offset % sampling_rate))
            if self._precise_round(halt % sampling_rate) != 0:
                halt = self._precise_round(halt + (sampling_rate - halt % sampling_rate))

            while offset < halt:
                hist[offset].append(note)
                offset += sampling_rate

        return hist

    def _to_word(self, notes, rest_word="XREST") -> str:
        if len(notes) == 0:
            return rest_word

        ordered_notes = sorted(notes, key=lambda n: n.pitch.midi, reverse=True)
        word = '_'.join([note.name.lower() for note in ordered_notes])
        return word


    def _precise_round(self, val, precision=10):
        return round(val * precision) / precision

    def _save_score_words(self, scores, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        in_text = json.dumps(scores)
        open(path, 'w').write(in_text)

    def _load_score_words(self, file_name):
        raw_text = open(file_name, 'r').read()
        return json.loads(raw_text)


class ScoreToVec:
    '''
    Takes in text form of scores and produces a vector embedding model
    '''
    def __init__(self, scores: list = [], path: str = '', load=True, **kwargs):
        if os.path.exists(path) and load:
            self.embedding = self._load_model(path)
        else:
            self.embedding = self.train_model(scores, **kwargs)
            self.embedding.save(path)

    def train_model(self, score_texts,
                          size=32,
                          min_count=1,
                          window=4,
                          workers=4,
                          sg=1,
                          **kwargs):
        '''
        score_texts - word representation of score as generated by ScoreToWord
        '''
        model = Word2Vec(sentences=score_texts,
                         size=size,
                         min_count=min_count,
                         window=window,
                         workers=workers,
                         sg=sg, **kwargs)
        return model.wv

    def _load_model(self, path):
        return KeyedVectors.load(path)

    def decode(self, vector):
        '''
        vector - arbitrary input embedding
        Returns - the topn most similar words to that vector
        '''
        return self.embedding.similar_by_vector(vector, topn=1)[0][0]

    def vocab(self):
        return self.embedding.vocab
