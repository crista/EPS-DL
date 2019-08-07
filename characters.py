# -*- coding: utf-8 -*-
'''
# An implementation of deep learning for replacing non-alphanumeric characters with space

Input:  'h' 'e' 'l' 'l' 'o' ' ' 'w' 'o' 'r' 'l' 'd' '!' ' ' 'F' 'o' 'o' '-' 'B' 'a' 'r' ' ' 'h' 'e' 'r' 'e' '.'
                                                     v                   v                                   v
Output: 'h' 'e' 'l' 'l' 'o' ' ' 'w' 'o' 'r' 'l' 'd' ' ' ' ' 'F' 'o' 'o' ' ' 'B' 'a' 'r' ' ' 'h' 'e' 'r' 'e' ' '

'''  # noqa

from __future__ import print_function
from keras.models import Sequential
from keras import layers
from keras.utils import plot_model
import numpy as np
from six.moves import range
import string


class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one-hot integer representation
    + Decode the one-hot or integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        """Initialize character table.

        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C):
        """One-hot encode given character C.

        # Arguments
            C: character, to be encoded.
        """
        x = np.zeros(len(self.chars))
        x[self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        """Decode the given vector or 1D array to their character output.

        # Arguments
            x: A vector or a 2D array of probabilities or one-hot representations;
                or a vector of character indices (used with `calc_argmax=False`).
            calc_argmax: Whether to find the character index with maximum
                probability, defaults to `True`.
        """
        if calc_argmax:
            x = x.argmax(axis=-1)
        return self.indices_char[x]


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

# All the printable characters.
chars = string.printable
ctable = CharacterTable(chars)

# Parameters for the model and dataset.
TRAINING_SIZE = len(chars)

def generate_characters():
    questions = []
    expected = []
    for c in chars:
        query = c
        ans = ' ' if query in string.punctuation else query

        questions.append(query)
        expected.append(ans)
    print('Total addition questions:', len(questions))
    return questions, expected

print('Generating data...')
questions, expected = generate_characters()

print('Vectorization...')
x = np.zeros((len(questions), len(chars)), dtype=np.bool)
y = np.zeros((len(questions), len(chars)), dtype=np.bool)
for i, c in enumerate(questions):
    x[i] = ctable.encode(c)
for i, c in enumerate(expected):
    y[i] = ctable.encode(c)

# Shuffle (x, y) in unison as the later parts of x will almost all be larger
# digits.
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# Validation data is the same as training data.
(x_train, x_val) = x, x
(y_train, y_val) = y, y

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

BATCH_SIZE = 50

def model_ff1():
    print('Build model...')
    epochs = 400
    model = Sequential()
    model.add(layers.Dense(len(chars), input_shape=(len(chars), ), activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model, epochs, "ff1-{}b-{}ep".format(BATCH_SIZE, epochs)

def model_ff2():
    print('Build model...')
    epochs = 100
    model = Sequential()
    model.add(layers.Dense(len(chars), input_shape=(len(chars), )))
    model.add(layers.Dense(len(chars), activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model, epochs, "ff2-{}b-{}ep".format(BATCH_SIZE, epochs)

def model_ff3():
    print('Build model...')
    epochs = 80
    model = Sequential()
    model.add(layers.Dense(len(chars), input_shape=(len(chars), )))
    model.add(layers.Dense(len(chars)))
    model.add(layers.Dense(len(chars), activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model, epochs, "ff3-{}b-{}ep".format(BATCH_SIZE, epochs)

model, epochs, name = model_ff1()

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
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q, '    T', correct, '    G', guess)

model.summary()
model.save(name + '.h5')
plot_model(model, to_file=name + '.png', show_shapes=True)