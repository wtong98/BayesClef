import metrics
import sys
import json
from internal.music2vec import ScoreFetcher, ScoreToWord, ScoreToVec
from internal.instantiate import *

SCORE_WORD_PATH = r'data/score_word_cache.json'

file_name = sys.argv[1]
my_score = json.loads(open(file_name, 'r').read())

illegal = ['<START>', '<END>']
token = [i.split('_') for i in my_score if not i in illegal]
# now play actual song
score = to_score(token, chord_with_ties_texture, duration=1)
score.show('musicxml')
