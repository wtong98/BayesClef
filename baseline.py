import sys
import datetime
import random
import os
import json
from internal.music2vec import ScoreFetcher, ScoreToWord, ScoreToVec

SCORE_WORD_PATH = r'data/score_word_cache.json'
OUTPUT_PATH = r'output/'

# Load in scores
print('Loading test scores...')
myScoreToWord = ScoreToWord(SCORE_WORD_PATH)
myScoreToWord.load_cache()

all_words = []
for i in myScoreToWord.scores:
    for wrd in i:
        all_words.append(wrd)

PIECE_LENGTH = 100
musical_piece = []
for i in range(PIECE_LENGTH):
    idx = random.randint(0, len(all_words)-1)
    musical_piece.append(all_words[idx])

# save output
if not os.path.isdir(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
open(OUTPUT_PATH + 'song_random_{}.json'.format(datetime.datetime.now()), 'w').write(json.dumps(musical_piece))
