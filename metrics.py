'''
Contains the metrics our system uses to evaluate the efficacy of our models
'''
from collections import Counter

from internal.music2vec import ScoreFetcher, ScoreToWord, ScoreToVec

SCORE_WORD_PATH = r'data/score_word_cache.json'

REST_SYMB = 'XREST'

def note_to_num(note):
    # converts a note such as f# to a number
    base = note[0]
    note_to_val = { 'c': 0, 'd': 1, 'e': 2, 'f': 3, 'g': 4, 'a': 5, 'b': 6 }
    val = note_to_val[base]
    if len(note) > 1:
        if note[1] == '-':
            val -= 1.5
        elif note[1] == '#':
            val += 0.5
    return val


timestep_dist_cache = {}
def timestep_dist(test, target):
    # Each distance of a full note results in a cost of 1 (i.e. D and F have cost of 1)
    global timestep_dist_cache
    if (test, target) in timestep_dist_cache: # serve from cache if available
        return timestep_dist_cache[(test, target)]

    notes_test = test.split('_')
    notes_target = target.split('_')
    target_vals = [note_to_num(j) for j in notes_target if not j == REST_SYMB]
    cost = None
    for note in notes_test:
        if not note == REST_SYMB and len(target_vals) > 0:
            note_val = note_to_num(note)
            if cost == None: # get cost ready for addition
                cost = 0
            cost += min([abs(val - note_val) for val in target_vals])
        else:
            if notes_test == notes_target: # i.e. both XREST
                cost = 0
    if cost == None: # happens if all of target was XREST
        return 3 # return cost of 3 in this case
    dist = cost/len(notes_test)
    timestep_dist_cache[(test, target)] = dist
    return dist

def chord_dist(test, target):
    # Each distance of a full note results in a cost of 1 (i.e. D and F have cost of 1)
    # and the cost is averaged by the number of notes in each chord
    # Note: Metric is not symmetric, meaning you are only penalized for distance of test
    # to nearest note in target and not the reverse
    distance = 0
    for i in range(len(test)):
        distance += timestep_dist(test[i], target[i])
    return distance

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

    # now compute metric on score
    for i in range(1, len(score)-k):
        timestep = tuple(score[i:i+k])
        # NOTE: More efficient approaches exist, but not employed for the sake
        # of prioritizing efforts
        min_cost = float('inf')
        for test in timesteps:
            cost = chord_dist(timestep, test)
            if cost < min_cost:
                min_cost = cost
        total_score += min_cost
        n += 1
    return (total_score/n)*100


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

