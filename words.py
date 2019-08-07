# -*- coding: utf-8 -*-
'''
# An implementation of deep learning for deetcting the N most frequenctly occurring integers
E.g. with N = 3
Input:  [10, 12, 10, 11, 2, 2, 2, 1, 1]
Output: [10, 2, 1] (Not necessarily in this order)

'''  # noqa

from __future__ import print_function
from keras.models import Sequential
from keras import layers
from keras.utils import plot_model
from keras.utils import to_categorical
import numpy as np
from six.moves import range
import string, re, collections

VOCAB_SIZE = 100
SAMPLE_SIZE = 100
TOP = 2

class WordTable(object):
    """Given a text file:
    + Encode the words to a one-hot integer representation
    + Decode the one-hot or integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, filename):
        """Initialize words table.

        # Arguments
            filename: The file from which to map the words.
        """
        global VOCAB_SIZE
        stopwords = set(open('stop_words.txt').read().split(','))
        self.all_words = re.findall('[a-z]{2,}', open(filename).read().lower())
        self.words = list(set([w for w in self.all_words if w not in stopwords]))
        self.word_indices = dict((w, i) for i, w in enumerate(self.words))
        self.indices_word = dict((i, w) for i, w in enumerate(self.words))

    def encode(self, W):
        """One-hot encode given word, or list of indices, W.

        # Arguments
            W: either a word or a list of indices, to be encoded.
        """
        if type(W) is string:
            x = np.zeros(VOCAB_SIZE)
            x[self.word_indices[W]] = 1
            return x
        elif type(W) is list: # Example: [3, 9, 5]
            x = np.zeros((SAMPLE_SIZE, VOCAB_SIZE))
            for i, w in enumerate(W):
                if i >= SAMPLE_SIZE: break
                x[i, w] = 1
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
            one_idxs = [i for i, v in enumerate(x) if v >= 0.5]
            return [self.indices_word[i] for i in one_idxs]
        elif x.ndim == 2: # a list of words, each one-hot encoded
            words = []
            for w in x:
                words.append(self.decode(w))
            return words
        else:
            raise Exception("Bad type to decode")


ctable = WordTable('pride-and-prejudice.txt')

# Parameters for the model and dataset.
TRAINING_SIZE = 50000

def generate_integer_list():
    questions = []
    expected = []
    for _ in range(0, TRAINING_SIZE):
#        query = [np.random.randint(0, VOCAB_SIZE) for _ in range(SAMPLE_SIZE)]

        # Grab a slice of the input file of size SAMPLE_SIZE
        index = np.random.randint(0, len(ctable.all_words) - SAMPLE_SIZE)
        querytmp = ctable.all_words[index:index+SAMPLE_SIZE]
        # Replace unknown words with known ones
        query = querytmp
        for i, w in enumerate(querytmp):
            if w not in ctable.words[:VOCAB_SIZE] and query[i] == w:
                # Replace ALL occurrences in query with the same replacement word
                other = ctable.words[np.random.randint(0, VOCAB_SIZE/2)]
                query = [other if v == w else v for v in query]

#        query = [w if w in ctable.words[:VOCAB_SIZE] else ctable.words[np.random.randint(0, VOCAB_SIZE/2)] for w in query]
        # Now replace them by their word indices
        query = [ctable.word_indices[w] for w in query]
        counts = collections.Counter(w for w in query)
        top = counts.most_common(TOP)
        ans = list(list(zip(*top))[0])

        questions.append(query)
        expected.append(ans)
    print('Total addition questions:', len(questions))
    return questions, expected

print('Generating data...')
questions, expected = generate_integer_list()

print('Vectorization...')
x = np.zeros((len(questions), SAMPLE_SIZE, VOCAB_SIZE), dtype=np.int)
y = np.zeros((len(questions), VOCAB_SIZE), dtype=np.int)
for i, q in enumerate(questions):
    x[i] = ctable.encode(q)
for i, a in enumerate(expected):
    y[i][a] = 1

#for i in range(2):
#    print(questions[i])
#    print(expected[i])
#    print(x[i])
#    print(y[i])

# Shuffle (x, y) in unison as the later parts of x will almost all be larger
# digits.
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# Explicitly set apart 10% for validation data that we never train over.
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

BATCH_SIZE = 32

def model_ff():
    print('Build model...')
    epochs = 200
    model = Sequential()
    model.add(layers.Dense(VOCAB_SIZE, activation='relu', input_shape=(SAMPLE_SIZE, VOCAB_SIZE)))
#    model.add(layers.Dense(SAMPLE_SIZE, activation='relu'))
#    model.add(layers.Dropout(0.5))
#    model.add(layers.Dense(150, activation='relu'))
    model.add(layers.Flatten())
#    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(VOCAB_SIZE, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model, epochs, "words-ff2-{}b-{}ep".format(BATCH_SIZE, epochs)

model, epochs, name = model_ff()
model.summary()
plot_model(model, to_file=name + '.png', show_shapes=True)

# Train the model each generation and show predictions against the validation
# dataset.
for iteration in range(1, epochs):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(x_val, y_val))
    # Select 10 samples from the validation set at random so we can visualize
    # errors.
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict(rowx, verbose=0)
        print(preds)
#        preds[preds>=0.5] = 1
#        preds[preds<0.5] = 0

        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0])
        print('T', correct, '    G', guess)
        print('G_count ', sum(v >= 0.5 for v in preds[0]), ' Guess_count ', len(guess))

model.summary()
model.save(name + '.h5')
