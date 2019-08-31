from keras import backend as K
import numpy as np
import string

#INPUT_SIZE = 75
#MAX_WORDS = 5
#MAX_WORD_SIZE = 20
#INPUT_VOCAB_SIZE = 100
#BATCH_SIZE = 1
#OUTPUT_SIZE = 5

INPUT_SIZE = 10
MAX_WORD_SIZE = 4
INPUT_VOCAB_SIZE = 12
BATCH_SIZE = 2
OUTPUT_SIZE = 5

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
        global INPUT_SIZE, MAX_WORDS_PER_SAMPLE, INPUT_VOCAB_SIZE
        # Input symbols
        self.characters = sorted(string.printable)
        self.char_indices = dict((c, i) for i, c in enumerate(self.characters))
        self.indices_char = dict((i, c) for i, c in enumerate(self.characters))
        max_chars = 0
        with open('pride-and-prejudice.txt') as f:
            for line in f:
                if len(line) > max_chars: max_chars = len(line)

        INPUT_SIZE = max_chars
        MAX_WORDS_PER_SAMPLE = int(max_chars/2)
        INPUT_VOCAB_SIZE = len(self.characters)

    def to_indices(self, symbols):
        return [self.char_indices[c] for c in symbols if c in string.printable]

    def from_indices(self, indices):
        return [self.indices_char[i] for i in indices]

    def encode_one_hot(self, S, typ="char"):
        """One-hot encode given a list of character indices, C.
        """
        if typ == "char": # Return a list of arrays
            all = []
            for s in S:
                x = np.zeros((INPUT_VOCAB_SIZE)) 
                x[s] = 1 
                all.append(x)
            return all
        else:
            x = np.zeros((MAX_WORDS_PER_SAMPLE, MAX_WORD_SIZE, INPUT_VOCAB_SIZE))
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
            #print(f'Top index is {one_idx} and value is ', x[one_idx])
            return self.indices_char[one_idx]
        elif x.ndim == 2: # a list of symbols, each one-hot encoded
            return ''.join([self.decode(c) for c in x])
        elif x.ndim == 3:
            words = [self.decode(w).strip() for w in x]
            return ' '.join(words)

        else:
            raise Exception("Bad type to decode")

ctable = SymbolTable()

def SpaceDetector(x):
    print("x-sh", x.shape)
#    print("input: ", K.eval(x))

    sp_idx = 0 #ctable.char_indices[' ']
    sp = np.zeros((INPUT_VOCAB_SIZE))
    print(sp)
    sp[sp_idx] = 1

    filtered = x * sp
#    print("filtered:", K.eval(filtered))
    sp_positions = K.tf.where(K.tf.equal(filtered, 1)) # row indices
    print(sp_positions.shape)
#    print("sp-p:", K.eval(sp_positions))

    starts = sp_positions[:-1] + [0, 1, 0]
    stops = sp_positions[1:] + [0, 0, INPUT_VOCAB_SIZE]
    sizes = stops - starts + [1, 0, 0]
    where = K.tf.equal(sizes[:, 0], 1)
    starts = K.tf.boolean_mask(starts, where) # Remove multi-sample rows
    sizes = K.tf.boolean_mask(sizes, where) # Same
    where = K.tf.greater(sizes[:, 1], 0)
    starts = K.tf.boolean_mask(starts, where) # Remove words with 0 length (consecutive spaces)
    sizes = K.tf.boolean_mask(sizes, where) # Same

    print("starts:", starts, "sh:", starts.shape)
    print("stops:", stops)
    print("sizes:", sizes, "sh:", sizes.shape)

    slices = K.map_fn(lambda info: K.tf.pad(K.squeeze(K.slice(x, info[0], info[1]), 0), [[0, MAX_WORD_SIZE - info[1][1]], [0,0]], "CONSTANT"), [starts, sizes], dtype=float)
    return slices

#with open('pride-and-prejudice.txt') as f:
#    lines = f.readlines()

#data = []
#for line in lines[0:BATCH_SIZE]:
#    if line.isspace(): continue
#    onehots = ctable.encode_one_hot(ctable.to_indices(list(' ' + line.strip() + ' ')))
#    data.append(onehots)
#    for _  in range(len(onehots), INPUT_SIZE):
#        data.append(np.zeros((INPUT_VOCAB_SIZE)))

#x = np.array(data)
#y = SpaceDetector(x)
#print(K.eval(y))

x = np.zeros((BATCH_SIZE, INPUT_SIZE, INPUT_VOCAB_SIZE))
x[0,0,0]=1 # space
x[0,1,1]=1 
x[0,2,3]=1
x[0,3,0]=1 # space
x[0,4,7]=1
x[0,5,9]=1
x[0,6,0]=1 # space
x[0,7,2]=1
x[0,8,4]=1
x[0,9,0]=1 # space

x[1,0,0]=1 # space
x[1,1,1]=1 
x[1,2,0]=1  # space
x[1,3,3]=1
x[1,4,7]=1
x[1,5,9]=1
x[1,6,0]=1 # space
x[1,7,2]=1
x[1,8,0]=1 # space
x[1,9,0]=1 # space

y=SpaceDetector(K.variable(x))
print(K.eval(y))