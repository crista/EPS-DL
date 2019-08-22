'''
# Count words in a file

**Summary of the algorithm**

- We start with input sequences of characters
- We produce words and their counts
'''
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras import layers, metrics
from keras import backend as K
from keras.utils import plot_model

import numpy as np
import sys, os, string

INPUT_SIZE = 100
OUTPUT_SIZE = 100
MAX_WORD_SIZE = 20
INPUT_VOCAB_SIZE = 80

file = 'pride-and-prejudice.txt' 
if len(sys.argv) > 1:
    data_folder = sys.argv[1]

class SymbolTable(object):
    """Given a text file:
    + Encode the characters to a one-hot integer representation
    + Decode the one-hot or integer representation to their word output
    """
    def __init__(self):
        """Initialize words table.
        # Arguments
            filename: The file from which to map the words.
        """
        global INPUT_SIZE, OUTPUT_SIZE, INPUT_VOCAB_SIZE
        # Input symbols
        self.characters = sorted(string.printable)
        self.char_indices = dict((c, i) for i, c in enumerate(self.characters))
        self.indices_char = dict((i, c) for i, c in enumerate(self.characters))
        max_chars = 0
        with open(file) as f:
            for line in f:
                if len(line) > max_chars: max_chars = len(line)

        INPUT_SIZE = max_chars
        OUTPUT_SIZE = int(max_chars/2)
        INPUT_VOCAB_SIZE = len(self.characters)

    def to_indices(self, symbols):
        return [self.char_indices[c] for c in symbols if c in string.printable]

    def from_indices(self, indices):
        return [self.indices_char[i] for i in indices]

    def encode_one_hot(self, S, typ="char"):
        """One-hot encode given a list of character indices, C.
        """
        if typ == "char":
            x = np.zeros((INPUT_SIZE, INPUT_VOCAB_SIZE))  
            for i, s in enumerate(S):
                x[i, s] = 1 
        else:
            x = np.zeros((OUTPUT_SIZE, MAX_WORD_SIZE, INPUT_VOCAB_SIZE))
            for i, w in enumerate(S):
                for j, c in enumerate(w):
                    idx = self.char_indices[c]
                    x[i, j, idx] = 1
        return x

    def decode(self, x):
        """Decode the given vector or 1D array to their symbolic output.
        # Arguments
            x: A vector or a 2D array of probabilities or one-hot representations;
                or a vector of symbol indices.
        """
        if type(x) == list:
            one_idxs = [np.argmax(h) for h in x]
            return ''.join([self.indices_char[one_idx] for one_idx in one_idxs if one_idx != 0])

        elif x.ndim == 1: # either a single symbol, one-hot encoded, or multiple symbols
            #one_idxs = [i for i, v in enumerate(x) if v >= 0.5]
            one_idx = np.argmax(x)
#            print(f'Top index is {one_idx} and value is ', x[one_idx])
            return self.indices_char[one_idx]
        elif x.ndim == 2: # a list of symbols, each one-hot encoded
            return ''.join([self.decode(c) for c in x])
        elif x.ndim == 3:
            words = [self.decode(w).strip() for w in x]
            return ' '.join(words)

        else:
            raise Exception("Bad type to decode")

ctable = SymbolTable()
# Test
#t1 = list("Hello World! Foo bar, I say")
#t1i = [ctable.char_indices[c] for c in t1]
#print(t1i)
#onehot = ctable.encode_one_hot(t1i, typ="char")
#print(ctable.decode(onehot))
#
#t2 = ['marks', 'strongly', 'scotch', 'head', 'head', 'careless', 'animal']
#onehot = ctable.encode_one_hot(t2, typ="word")
#print(ctable.decode(onehot))

print('Number of unique input tokens:', INPUT_VOCAB_SIZE)
print('Max sequence length for inputs:', INPUT_SIZE)
print('Max sequence length for outputs:', OUTPUT_SIZE)

def normalization_layer_set_weights(n_layer):
    wb = []
    b = np.zeros((INPUT_VOCAB_SIZE), dtype=np.float32)
    w = np.zeros((INPUT_VOCAB_SIZE, INPUT_VOCAB_SIZE), dtype=np.float32)
    for c in string.ascii_lowercase:
        i = ctable.char_indices[c]
        w[i, i] = 1
    # Change capitals to lower case
    for c in string.ascii_uppercase:
        i = ctable.char_indices[c]
        il = ctable.char_indices[c.lower()]
        w[i, il] = 1
    wb.append(w)
    wb.append(b)
    n_layer.set_weights(wb)
    return n_layer

def build_model():
    print('Build model...')
    n_layer = Dense(INPUT_VOCAB_SIZE)
    raw_inputs = []
    normalized_outputs = []
    for _ in range(0, INPUT_SIZE):
        input_char = Input(shape=(INPUT_VOCAB_SIZE, ))
        filtered_char = n_layer(input_char)
        raw_inputs.append(input_char)
        normalized_outputs.append(filtered_char)
    normalization_layer_set_weights(n_layer)

    merged_output = layers.concatenate(normalized_outputs, axis=-1)

    model = Model(inputs=raw_inputs, outputs=normalized_outputs)

    return model


model = build_model()
model.summary()
plot_model(model, to_file='tf-no-learning.png', show_shapes=True)
with open(file) as f:
    for line in f:
        onehot = ctable.encode_one_hot(ctable.to_indices(list(line)))
        inputs = []
        for a in onehot:
            x = np.zeros((1, INPUT_VOCAB_SIZE))
            x[0] = a
            inputs.append(x)

        preds = model.predict(inputs, batch_size=1)
        print(len(preds))
        print("Input:", ctable.decode(onehot))
        print("Output-raw:", preds[:2])
        print("Output:", ctable.decode(preds))
        raise Exception("foo")
#        wf = count_words(onehot)
#        for w, f in wf.items():
#            print(w, "-", f)


