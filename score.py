import metrics
import sys
import json
from internal.music2vec import ScoreFetcher, ScoreToWord, ScoreToVec

SCORE_WORD_PATH = r'data/score_word_cache.json'

file_name = sys.argv[1]
my_score = json.loads(open(file_name, 'r').read())

# Load in scores
print('Loading test scores...')
myScoreToWord = ScoreToWord(SCORE_WORD_PATH)
myScoreToWord.load_cache()

# Test chord_similarity metric
print('Testing chord_similarity...')
cost = metrics.chord_similarity(my_score, myScoreToWord.scores)
print('chord_similarity', cost)

