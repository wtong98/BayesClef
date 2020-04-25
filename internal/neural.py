import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time
import datetime
import os
import random
import json
import io

class GRUTypeNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1, drop_prob=0.2):
        super(GRUTypeNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_size = input_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim+1, hidden_dim, n_layers, batch_first=True)
        # 3 linear layers between GRU and softmax
        self.fc_deep0 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim+1)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        #out = self.softmax(self.fc(self.relu(out[:,-1])))
        fc0_out = self.relu(self.fc_deep0(self.relu(out)))
        out = F.log_softmax(self.fc(fc0_out), dim=-1)
        return out, h

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim)

    def _epoch_loader(self):
        # first shuffle data
        random.shuffle(self.training_examples)
        accumulator = ([], []) # accumulates a batch
        for i in self.training_examples:
            if len(accumulator[0]) >= self.batch_size:
                yield accumulator
                accumulator = ([], [])
            inp = []
            for idx in range(len(i[0])):
                # expand to one-hot
                inp.append([1. if i[0][idx] == q-1 else 0. for q in range(self.input_size+1)])
            accumulator[0].append(inp)
            accumulator[1].append(i[1])

    def _get_train_loader(self, type_sequences, batch_size, seq_len):
        all_trains = []
        for seq in type_sequences:
            seq_mod = [-1] + list(seq) # add start token
            for i in range(len(seq_mod)-seq_len):
                all_trains.append((seq_mod[i:i+seq_len],seq_mod[i+1:i+seq_len+1]))
        self.training_examples = all_trains
        self.batch_size = batch_size
        return self._epoch_loader

    def fit(self, type_sequences, learn_rate=0.001, EPOCHS=10, SEQ_LEN=30):
        # Defining loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate)

        # Getting train loader ready
        train_loader = self._get_train_loader(type_sequences, batch_size=20, seq_len=SEQ_LEN)

        self.train()
        print("Starting Training of type model")
        epoch_times = []
        # Start training loop
        for epoch in range(1,EPOCHS+1):
            start_time = time.time()
            avg_loss = 0.
            counter = 0
            for x_batch, label_batch in train_loader():
                h = self.init_hidden(self.batch_size)
                x = torch.tensor(x_batch)
                label = torch.tensor(label_batch)
                counter += 1
                self.zero_grad()

                out, h = self(x, h)
                loss = 0
                for i, lab in enumerate(label):
                    loss += criterion(out[i], lab)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                if counter%400 == 0:
                    print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, int(len(self.training_examples)/self.batch_size), avg_loss/counter))
            current_time = time.time()
            print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss/self.batch_size))
            print("Total Time Elapsed: {} seconds".format(str(current_time-start_time)))
            epoch_times.append(current_time-start_time)
        print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

    def sample(self, length, top_k=2):
        type_seq = []
        start_vec = torch.tensor([[[1. if -1 == q-1 else 0. for q in range(self.input_size+1)]]])
        should_stop = False
        curr_vec = start_vec
        curr_h = None # Starting hidden state
        while not should_stop:
            out, curr_h = self(curr_vec, curr_h)
            softmax_out = list(out[0][-1]) # get prediction for final output
            top_k_out = np.argsort(softmax_out)[-top_k:]
            rand_idx = random.randint(0, top_k - 1)
            choice = top_k_out[rand_idx]
            type_seq.append(choice)
            # now update vector
            curr_vec = torch.tensor([[[1. if choice == q-1 else 0. for q in range(self.input_size+1)]]])

            if len(type_seq) >= length:
                should_stop = True
        return np.array(type_seq), None
