import json
import os
import random
import os.path
import pickle
from collections import defaultdict

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from music21 import corpus
from music21 import interval
from music21 import note
from tqdm import tqdm

import numpy as np

# Global vars
START_WORD = '<START>'
END_WORD = '<END>'
REST_WORD = '<REST>'

class ScoreFetcher:
    '''
    Load scores
    '''
    def __init__(self, path: str = '', load=True):
        self.save_path = path

    def fetch(self):
        self.scores = self.query_scores()
        self._save_score_cache()

    @staticmethod
    def query_scores(artist='bach', debug=False):
        bundle = corpus.search(artist, 'composer')
        if debug:
            print('Loading {0} scores'.format(len(bundle)))
        return [metadata.parse() for metadata in bundle]

    def load_cache(self):
        with open(self.save_path, 'rb') as fp:
            self.scores = pickle.load(fp)

    def _save_score_cache(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'wb') as fp:
            pickle.dump(self.scores, fp)

class ScoreToWord:
    '''
    Convert scores into word form
    '''

    def __init__(self, path: str = ''):
        self.save_path = path

    def process(self, raw_scores, test_split=0.05):
        all_scores = list(tqdm(self.scores_to_text(raw_scores)))
        idxes = list(range(len(all_scores)))
        random.shuffle(idxes)
        cutoff = int(len(all_scores)*test_split)
        test_idxes = idxes[:cutoff]
        train_idxes = idxes[cutoff:]
        self.test_scores = [all_scores[i] for i in test_idxes]
        self.scores = [all_scores[i] for i in train_idxes]
        print('Number of scores: {} with {} test scores'.format(len(self.scores), len(self.test_scores)))
        self._save_score_words({ 'scores': self.scores, 'test_scores': self.test_scores }, self.save_path)

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
        full_text = [START_WORD] + text + [END_WORD]
        return full_text

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

    def _to_word(self, notes) -> str:
        if len(notes) == 0:
            return REST_WORD

        ordered_notes = sorted(notes, key=lambda n: n.pitch.midi, reverse=True)
        word = '_'.join([note.name.lower() for note in ordered_notes])
        return word


    def _precise_round(self, val, precision=10):
        return round(val * precision) / precision

    def _save_score_words(self, scores, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        in_text = json.dumps(scores)
        open(path, 'w').write(in_text)

    def load_cache(self):
        raw_text = open(self.save_path, 'r').read()
        data = json.loads(raw_text)
        self.scores = data['scores']
        self.test_scores = data['test_scores']


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
