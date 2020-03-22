##########################################################
# LSTM Generator
# This model serves as a benchmark with which to base
# our model performance off of
##########################################################
import torch
import torch.nn as nn
import numpy as np
import io

from internal.music2vec import ScoreFetcher, ScoreToWord, ScoreToVec
from internal.type_models import BayesianGaussianTypeModel
import os.path
import pickle
import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm
from music21 import converter

SCORE_PATH = r'data/score_cache.json'
SCORE_WORD_PATH = r'data/score_word_cache.json'
EMBEDDING_PATH = r'data/embedding.wv'

BATCH_SIZE = 10
SEQ_SIZE = 10
LSTM_SIZE = 10
EMBEDDING_SIZE = 32
N_VOCAB = None # to update
GRADIENTS_NORM = 5

# Prepare for saving
no_object_computes = True
def should_compute(obj : object, prompt: str = 'Load cache'):
    global no_object_computes
    if no_object_computes:
        inp = input(prompt + ' [y/n]?')
        if len(inp) > 0 and inp[0] == 'n':
            no_object_computes = False
            return True
        obj.load_cache()
        return False
    else:
        return True

myScoreToWord = ScoreToWord(SCORE_WORD_PATH)
myScoreToWord.load_cache()

print('Training embedding model...')
score_word_to_vec = ScoreToVec(myScoreToWord.scores, path=EMBEDDING_PATH)

N_VOCAB = len(score_word_to_vec.vocab())
# Set up for one hot encoding
vocab = list(score_word_to_vec.vocab().keys())
vocab_to_int = {}
int_to_vocab = {}
for i,val in enumerate(vocab):
    vocab_to_int[val] = i
    int_to_vocab[i] = val

###############
# Process data
###############
chunk_size = []
X_train = []
y_train = []
for score in myScoreToWord.scores:
    score_vectors = [score_word_to_vec.embedding[i] for i in score]
    score_ints = [vocab_to_int[i] for i in score]
    # TODO: Add START and STOP tokens to beginning and end
    X_train += [[[i] for i in score_vectors[:-1]]]
    y_train += [[[i] for i in score_ints[1:]]]

class Ticker:
    score_idx = 0
    place_idx = 0

# Will keep track of how far along we are in batch generation
ticker = Ticker()
def get_batches():
    global BATCH_SIZE, SEQ_SIZE, ticker
    for i in range(BATCH_SIZE):
        if ticker.place_idx > len(X_train[ticker.score_idx]) - SEQ_SIZE + 1:
            ticker.place_idx = 0
            ticker.score_idx += 1
        if ticker.score_idx > len(X_train):
            ticker.score_idx = 0
        yield X_train[ticker.score_idx][ticker.place_idx:ticker.place_idx + SEQ_SIZE], \
                    y_train[ticker.score_idx][ticker.place_idx:ticker.place_idx + SEQ_SIZE]
        ticker.place_idx += 1


#####
# Set up generator
#####
def predict(device, net, words, n_vocab, vocab_to_int, int_to_vocab, top_k=5):
    net.eval()

    state_h, state_c = net.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        #ix = torch.tensor([[vocab_to_int[w]]]).to(device)
        ix = torch.tensor([[score_word_to_vec.embedding[w]]])
        output, (state_h, state_c) = net(ix, (state_h, state_c))

    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])
    for _ in range(100):
        #ix = torch.tensor([[choice]]).to(device)
        ix = torch.tensor([[score_word_to_vec.embedding[int_to_vocab[choice]]]])
        output, (state_h, state_c) = net(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(int_to_vocab[choice])

    print(words)


##################
# Set up network
# Config inspired by: https://machinetalk.org/2019/02/08/text-generation-with-pytorch/
##################
class RNNModule(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size):
        super(RNNModule, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.lstm = nn.LSTM(embedding_size,
                            lstm_size,
                            batch_first=True)
        self.dense = nn.Linear(lstm_size, n_vocab)

    def forward(self, x, prev_state):
        output, state = self.lstm(x, prev_state)
        logits = self.dense(output)

        return logits, state

    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))

def get_loss_and_train_op(net, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    return criterion, optimizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = RNNModule(N_VOCAB, SEQ_SIZE,
                EMBEDDING_SIZE, LSTM_SIZE)
net = net.to(device)

criterion, optimizer = get_loss_and_train_op(net, 0.01)

iteration = 0

# Train it!
for e in range(50):
    batches = get_batches()
    state_h, state_c = net.zero_state(BATCH_SIZE)

    # Transfer data to GPU
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for x, y in batches:
        iteration += 1

        # Tell it we are in training mode
        net.train()

        # Reset all gradients
        optimizer.zero_grad()

        # Transfer data to GPU
        x = torch.tensor(x).to(device)
        y = torch.tensor(y).to(device)

        logits, (state_h, state_c) = net(x, (state_h, state_c))
        #predicted = [i[0] for i in logits]
        #predicted = torch.tensor(predicted).to(device)
        predicted = logits
        #y_inp = [i[0] for i in y]
        #y_inp = torch.tensor(y_inp).to(device)
        y_inp = y
        loss = criterion(logits.transpose(1, 2), y)
        #loss = criterion(predicted, y_inp)

        state_h = state_h.detach()
        state_c = state_c.detach()

        # loss_val = loss.item()
        print('loss', loss.item())


        # Perform back-propagation
        loss.backward()

        _ = torch.nn.utils.clip_grad_norm_(
            net.parameters(), GRADIENTS_NORM)

        # Update the network's parameters
        optimizer.step()

        if iteration % 100 == 0:
            predict(device, net, ['e_e_a_c'], N_VOCAB,
                    vocab_to_int, int_to_vocab, top_k=5)
            print('Epoch: {}/{}'.format(e, 200),
                  'Iteration: {}'.format(iteration),
                  'Loss: {}'.format(loss))

        if iteration % 1000 == 0:
            predict(device, net, ['e_e_a_c'], n_vocab,
                    vocab_to_int, int_to_vocab, top_k=5)
            torch.save(net.state_dict(),
                       'checkpoint_pt/model-{}.pth'.format(iteration))
