'''
Contains the metrics our system uses to evaluate the efficacy of our models
'''
from collections import Counter

from internal.music2vec import ScoreFetcher, ScoreToWord, ScoreToVec

SCORE_WORD_PATH = r'data/score_word_cache.json'

def chord_similarity(score, test_scores, k=4):
    '''
    Overview:
    For every k-timestep sequence in the score
    we find the nearest timestep in the test score set and compute
    the difference with each full note difference having a cost of 1.
    This approach can be thought of as finding the nearest chord in the test
    set, with the assumption being that, a genre of music tends to have common
    chords and we can use this to evaluate how well our model has learned
    without unfairly assigning cost to creative differences in how the chords
    are put together in sequence.
    '''
    # First compute all the unique k-timestep sequences
    # these are essentially equivalent to k note chords
    timesteps = Counter()
    for sc in test_scores:
        for i in range(1, len(sc)-k): # note: cuts off <START> and <END> tags
            key = tuple(sc[i:i+k])
            timesteps[key] += 1
    total_score = 0
    n = 0
    for i in range(1, len(score)-k):
        timestep = tuple(score[i:i+k])
        if not timestep in timesteps:
            # TODO: Compute actual note distance as opposed to flat cost of 1
            # if not exact match
            total_score += 1
        n += 1
    return total_score/n


if __name__ == '__main__':
    # Load in scores
    print('Loading in scores...')
    myScoreToWord = ScoreToWord(SCORE_WORD_PATH)
    myScoreToWord.load_cache()

    # Test chord_similarity metric
    print('Testing chord_similarity...')
    test_score = myScoreToWord.scores[0]
    cost = chord_similarity(test_score, myScoreToWord.scores[1:])
    print('chord_similarity', cost)

