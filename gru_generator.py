import torch
import torch.nn as nn
import numpy as np
import time
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
vocab_to_int = {}
int_to_vocab = {}
vocab = list(score_word_to_vec.vocab().keys())
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
    y_train += [[i for i in score_ints[1:]]]

class Ticker:
    score_idx = 0
    place_idx = 0

# Will keep track of how far along we are in batch generation
ticker = Ticker()
def get_batches():
    global BATCH_SIZE, SEQ_SIZE, ticker, vocab_to_int

    for i in range(BATCH_SIZE):
        # Handling looping
        if ticker.place_idx > len(X_train[ticker.score_idx]) - SEQ_SIZE + 1:
            ticker.place_idx = 0
            ticker.score_idx += 1
        if ticker.score_idx > len(X_train):
            ticker.score_idx = 0

        x_batch = X_train[ticker.score_idx][ticker.place_idx:ticker.place_idx + SEQ_SIZE]
        label_category = y_train[ticker.score_idx][ticker.place_idx:ticker.place_idx + SEQ_SIZE]

        # Now handle if unable to complete because reached end of score
        if len(x_batch) < SEQ_SIZE:
            deficit = SEQ_SIZE - len(x_batch)
            x_batch += [[score_word_to_vec.embedding['<END>']] for i in range(deficit)]
            label_category += [vocab_to_int['<END>'] for i in range(deficit)]

        yield torch.tensor(x_batch), \
                    torch.tensor(label_category)
        ticker.place_idx += 1


is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Core GRU code from: https://blog.floydhub.com/gru-with-pytorch/
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.softmax(self.fc(self.relu(out[:,-1])))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

def train(train_loader, learn_rate, hidden_dim=256, EPOCHS=20, model_type="GRU"):
    # Setting common hyperparameters
    input_dim = EMBEDDING_SIZE
    output_dim = N_VOCAB
    n_layers = 2
    # Instantiating the models
    if model_type == "GRU":
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    model.to(device)

    # Defining loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    print("Starting Training of {} model".format(model_type))
    epoch_times = []
    # Start training loop
    for epoch in range(1,EPOCHS+1):
        start_time = time.time()
        h = model.init_hidden(BATCH_SIZE)
        avg_loss = 0.
        counter = 0
        for x, label in train_loader():
            counter += 1
            if model_type == "GRU":
                h = h.data
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            #print([[float(i) for i in list(i)] for i in list(out)])
            loss = criterion(out, label.to(device).long())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if counter%200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), avg_loss/counter))
        current_time = time.time()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss/BATCH_SIZE))
        print("Total Time Elapsed: {} seconds".format(str(current_time-start_time)))
        epoch_times.append(current_time-start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    return model

def evaluate(model, test_x, test_y, label_scalers):
    model.eval()
    outputs = []
    targets = []
    start_time = time.clock()
    for i in test_x.keys():
        inp = torch.from_numpy(np.array(test_x[i]))
        labs = torch.from_numpy(np.array(test_y[i]))
        h = model.init_hidden(inp.shape[0])
        out, h = model(inp.to(device).float(), h)
        outputs.append(label_scalers[i].inverse_transform(out.cpu().detach().numpy()).reshape(-1))
        targets.append(label_scalers[i].inverse_transform(labs.numpy()).reshape(-1))
    print("Evaluation Time: {}".format(str(time.clock()-start_time)))
    sMAPE = 0
    for i in range(len(outputs)):
        sMAPE += np.mean(abs(outputs[i]-targets[i])/(targets[i]+outputs[i])/2)/len(outputs)
    print("sMAPE: {}%".format(sMAPE*100))
    return outputs, targets, sMAPE

lr = 0.001
gru_model = train(get_batches, lr, model_type="GRU")