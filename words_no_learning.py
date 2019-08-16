# -*- coding: utf-8 -*-
'''
# An implementation of deep learning for counting symbols
Input:  [10, 12, 10, 11, 2, 2, 2, 1, 1]
Output: words=[2, 10, 1, 12, 11] counts=[3, 2, 2, 1, 1] (Not necessarily in this order)

'''  # noqa

from __future__ import print_function
from keras.models import Sequential, Model
from keras import layers, metrics
from keras import backend as K
from keras.utils import plot_model
from keras.utils import to_categorical
import numpy as np
from six.moves import range
import string, re, collections, os, sys

# Parameters for the model and dataset.
TRAINING_SIZE = 50000
VOCAB_SIZE = 1000
SAMPLE_SIZE = 100
TOP = 2
BATCH_SIZE = 50

BIN_SIZE = 8

data_folder = 'words_data' 
if len(sys.argv) > 1:
    data_folder = data_folder + '_' + sys.argv[1]
train_x = os.path.join(data_folder, 'train_x.txt')
train_y = os.path.join(data_folder, 'train_y.txt')
val_x = os.path.join(data_folder, 'val_x.txt')
val_y = os.path.join(data_folder, 'val_y.txt')


class WordTable(object):
    """Given a text file:
    + Encode the words to a one-hot integer representation
    + Decode the one-hot or integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self):
        """Initialize words table.

        # Arguments
            filename: The file from which to map the words.
        """
        global TRAINING_SIZE
        global VOCAB_SIZE
        global SAMPLE_SIZE
        global BATCH_SIZE

        self.words = set()
        nlines = 0
        max_words = 0
        with open(train_x) as f:
            for line in f:
                words = line.split()
                self.words.update(words)

                nlines = nlines + 1
                if max_words < len(words):
                    max_words = len(words)

        self.words = list(sorted(self.words))
        self.word_indices = dict((w, i) for i, w in enumerate(self.words))
        self.indices_word = dict((i, w) for i, w in enumerate(self.words))

        TRAINING_SIZE = nlines
        VOCAB_SIZE = len(self.words)
        SAMPLE_SIZE = max_words
        BATCH_SIZE = 50

    def words_to_indices(self, words):
        return [self.word_indices[w] for w in words]

    def indices_to_words(self, indices):
        return [self.indices_word[i] for i in indices]

    def encode_one_hot(self, W, forConv=False):
        """One-hot encode given word, or list of indices, W.

        # Arguments
            W: either a word or a list of indices, to be encoded.
        """
        if type(W) is string:
            x = np.zeros(VOCAB_SIZE)
            x[self.word_indices[W]] = 1
            return x
        elif type(W) is list: # Example: [3, 9, 5]
            x = np.zeros((SAMPLE_SIZE, VOCAB_SIZE))  if not forConv else np.zeros((SAMPLE_SIZE, VOCAB_SIZE, 1))
            for i, w in enumerate(W):
                if i >= SAMPLE_SIZE: break
                if not forConv:
                    x[i, w] = 1 
                else:
                    x[i, w, 0] = 1
            return x
        else:
            raise Exception("Bad type to encode")

    def encode_binary(self, W):
        if type(W) is string:
            x = np.zeros(BIN_SIZE)
            for n in range(8): 
                n2 = pow(2, n)
                x[n] = 1 if (self.word_indices[W] & n2) == n2 else 0
            return x
        elif type(W) is list: # Example: [3, 9, 5] (indices already)
            x = np.zeros((SAMPLE_SIZE, BIN_SIZE, 1))
            for i, w in enumerate(W):
                if i >= SAMPLE_SIZE: break
                for n in range(BIN_SIZE): 
                    n2 = pow(2, n)
                    x[i, n, 0] = 1 if (w & n2) == n2 else 0
            return x
        else:
            raise Exception("Bad type to encode")


    def decode(self, x):
        """Decode the given vector or 1D array to their character output.

        # Arguments
            x: A vector or a 2D array of probabilities or one-hot representations;
                or a vector of word indices (used with `calc_argmax=False`).
            calc_argmax: Whether to find the word index with maximum
                probability, defaults to `True`.
        """
        if x.ndim == 1: # either a single word, one-hot encoded, or multiple words
            #one_idxs = [i for i, v in enumerate(x) if v >= 0.5]
            one_idxs = np.argpartition(x, -TOP)[-TOP:]
            print(f'Top 2 indices are {one_idxs} and values are ', np.rint(x[one_idxs]))
            return [self.indices_word[i] for i in one_idxs]
        elif x.ndim == 2: # a list of words, each one-hot encoded
            words = []
            for w in x:
                words.append(self.decode(w))
            return words
        else:
            raise Exception("Bad type to decode")


ctable = WordTable()
print(f'Words table with training size {TRAINING_SIZE}, batch size {BATCH_SIZE}, vocab size {VOCAB_SIZE} and sample size {SAMPLE_SIZE}')


def line_x_to_indices(line):
    words = line.split()
    return ctable.words_to_indices(words)

def line_y_to_indices(line):
    pairs = line.split(',')
    if len(pairs[0]) < 2: # no counts here
        return list(zip(ctable.words_to_indices(pairs), [1 for _ in range(len(pairs))]))
    else:
        words =  [p.split()[0] for p in pairs]
        counts = [int(p.split()[1]) for p in pairs] 
        w_indices = ctable.words_to_indices(words)
        return w_indices, counts 

def input_generator(nsamples, train=True, forConv=False):
    print('Generating input for ', 'training' if train else 'validation')
    f_x, f_y = (train_x, train_y) if train else (val_x, val_y)
    with open(f_x) as fx, open(f_y) as fy:
        j = 0
        x = np.zeros((nsamples, SAMPLE_SIZE, VOCAB_SIZE), dtype=np.int) if not forConv else np.zeros((nsamples, SAMPLE_SIZE, BIN_SIZE, 1), dtype=np.int)
        y = np.zeros((nsamples, VOCAB_SIZE), dtype=np.float64)
        for line_x, line_y in zip(fx, fy):
            question = line_x_to_indices(line_x)
            expected_w, expected_c = line_y_to_indices(line_y)
            #x[j] = ctable.encode_one_hot(question, forConv)
            x[j] = ctable.encode_binary(question)
            y[j][expected_w] = expected_c
            j = j + 1
            if j % nsamples == 0:
                yield x, y
                j = 0
                x = np.zeros((nsamples, SAMPLE_SIZE, VOCAB_SIZE), dtype=np.int) if not forConv else np.zeros((nsamples, SAMPLE_SIZE, BIN_SIZE, 1), dtype=np.int)
                y = np.zeros((nsamples, VOCAB_SIZE), dtype=np.float64)
        print("End of ", 'training' if train else 'validation')
        return x, y

def set_weights(clayer):
    wb = []
    b = np.zeros((100), dtype=np.float32)
    w = np.zeros((1, BIN_SIZE, 1, VOCAB_SIZE), dtype=np.float32)
    for i in range(VOCAB_SIZE):
        for n in range(BIN_SIZE):
            n2 = pow(2, n)
            w[0][n][0][i] = 1 if (i & n2) == n2 else -(BIN_SIZE-1)
    for i in range(VOCAB_SIZE):
        slice_1 = w[0, :, 0, i]
        n_ones = len(slice_1[ slice_1 == 1 ])
        if n_ones > 0: slice_1[ slice_1 == 1 ] = 1./n_ones  
    wb.append(w)
    wb.append(b)
    clayer.set_weights(wb)

def SumPooling2D(x):
    s = K.sum(x, axis=1) 
    return s

def model_convnet2D():
    print('Build model...')
    model = Sequential()
    model.add(layers.Conv2D(VOCAB_SIZE, (1, BIN_SIZE),  input_shape=(SAMPLE_SIZE, BIN_SIZE, 1)))
    set_weights(model.layers[0])
    model.add(layers.ReLU(threshold=1-1/BIN_SIZE))
    model.add(layers.Lambda(SumPooling2D))
    model.add(layers.Reshape((VOCAB_SIZE,)))

    return model, "words-nolearning-{}v-{}f".format(VOCAB_SIZE, BIN_SIZE)


model, name = model_convnet2D()
model.summary()
plot_model(model, to_file=name + '.png', show_shapes=True)

# Train the model each generation and show predictions against the validation
# dataset.
val_gen_2 = input_generator(20, train=False, forConv=True)
batch_x, batch_y = next(val_gen_2)
for i in range(len(batch_x)):
    intermediate_model = Model(inputs=model.input, outputs=[l.output for l in model.layers])
    preds = intermediate_model.predict(batch_x) # outputs a list of 4 arrays
    query = batch_x[i]
    expected = batch_y[i]
    prediction = preds[-1][i] # -1 is the output of the last layer

    correct = ctable.decode(expected)
    guess = ctable.decode(prediction)
    print('T', correct, '    G', guess)

