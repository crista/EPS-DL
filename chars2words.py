'''
#Sequence of characters to sequence of words.

**Summary of the algorithm**

- We start with input sequences from a domain (sequences of characters
    and corresponding target sequences from another domain
    (sequences of words).
- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).
- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    It uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.
- In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.
'''
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.utils import Sequence
import numpy as np
import sys, os, string

TRAINING_SIZE = 10000
INPUT_SIZE = 100
OUTPUT_SIZE = 100
INPUT_VOCAB_SIZE = 80
OUTPUT_VOCAB_SIZE = 20
BATCH_SIZE = 64
latent_dim = 256  # Latent dimensionality of the encoding space.

#batch_size = 64  # Batch size for training.
#epochs = 100  # Number of epochs to train for.
#num_samples = 10000  # Number of samples to train on.

# Path to the data txt file on disk.
#data_path = 'fra-eng/fra.txt'
data_folder = 'c2w_data' 
if len(sys.argv) > 1:
    data_folder = data_folder + '_' + sys.argv[1]
    VOCAB_SIZE = int(sys.argv[1])
train_x = os.path.join(data_folder, 'train_x.txt')
train_y = os.path.join(data_folder, 'train_y.txt')
val_x = os.path.join(data_folder, 'val_x.txt')
val_y = os.path.join(data_folder, 'val_y.txt')

class SymbolTable(object):
    """Given a text file:
    + Encode the characters to a one-hot integer representation
    + Decode the one-hot or integer representation to their char output
    """
    def __init__(self):
        """Initialize words table.
        # Arguments
            filename: The file from which to map the words.
        """
        global TRAINING_SIZE, INPUT_SIZE, OUTPUT_SIZE, INPUT_VOCAB_SIZE, OUTPUT_VOCAB_SIZE, BATCH_SIZE

        # Input symbols
        self.characters = sorted(string.printable)
        self.char_indices = dict((c, i) for i, c in enumerate(self.characters))
        self.indices_char = dict((i, c) for i, c in enumerate(self.characters))
        nlines = 0
        max_chars = 0
        with open(train_x) as f:
            for line in f:
                if len(line) > max_chars: max_chars = len(line)

        # Output symbols
        self.words = set()
        nlines = 0
        max_words = 0
        with open(train_y) as f:
            for line in f:
                ws = line.strip().split(',')
                self.words.update(ws)
                nlines = nlines + 1
                if max_words < len(ws):
                    max_words = len(ws)
        with open(val_y) as f:
            for line in f:
                ws = line.strip().split(',')
                self.words.update(ws)
                if max_words < len(ws):
                    max_words = len(ws)

        self.words = sorted(list(self.words))
        self.word_indices = dict((w, i+1) for i, w in enumerate(self.words))
        self.indices_word = dict((i+1, w) for i, w in enumerate(self.words))
        self.word_indices['@start@'] = 0
        self.indices_word[0] = '@start@'
        self.word_indices['@end@'] = OUTPUT_VOCAB_SIZE + 1
        self.indices_word[OUTPUT_VOCAB_SIZE + 1] = '@end@'

        TRAINING_SIZE = nlines
        INPUT_SIZE = max_chars
        OUTPUT_SIZE = max_words + 2
        INPUT_VOCAB_SIZE = len(self.characters)
        OUTPUT_VOCAB_SIZE = len(self.words) + 2
        print(self.words)

    def to_indices(self, symbols, typ="word"):
        if typ == "word": return [self.word_indices[w] for w in symbols]
        else: return [self.char_indices[c] for c in symbols if c in string.printable]

    def from_indices(self, indices, typ="word"):
        if typ == "word": return [self.indices_word[i] for i in indices]
        else: return [self.indices_char[i] for i in indices]

    def encode_one_hot(self, S, typ="word"):
        """One-hot encode given a list of character indices, C.
        """
        if typ == "word": x = np.zeros((OUTPUT_SIZE, OUTPUT_VOCAB_SIZE))  
        else: x = np.zeros((INPUT_SIZE, INPUT_VOCAB_SIZE))  
        for i, s in enumerate(S):
            x[i, s] = 1 
        return x

    def decode(self, x, typ="word"):
        """Decode the given vector or 1D array to their symbolic output.
        # Arguments
            x: A vector or a 2D array of probabilities or one-hot representations;
                or a vector of symbol indices.
        """
        if x.ndim == 1: # either a single symbol, one-hot encoded, or multiple symbols
            #one_idxs = [i for i, v in enumerate(x) if v >= 0.5]
            one_idx = np.argmax(x)
#            print(f'Top index is {one_idx} and value is ', x[one_idx])
            return self.indices_word[one_idx] if typ == "word" else self.indices_char[one_idx]
        elif x.ndim == 2: # a list of symbols, each one-hot encoded
            return [self.decode(s, typ) for s in x]
        else:
            raise Exception("Bad type to decode")

ctable = SymbolTable()

print('Number of samples:', TRAINING_SIZE)
print('Number of unique input tokens:', INPUT_VOCAB_SIZE)
print('Number of unique output tokens:', OUTPUT_VOCAB_SIZE)
print('Max sequence length for inputs:', INPUT_SIZE)
print('Max sequence length for outputs:', OUTPUT_SIZE)

def line_x_to_indices(line):
    return ctable.to_indices(list(line), typ="char")
    print(line)

def line_y_to_indices(line):
    words = ['@start@'] + line.split(',') + ['@end@'] 
    return ctable.to_indices(words)

class DataGenerator(Sequence):
    def __init__(self, f_x, f_y, batch_size, size, shuffle=True):
        self.size = size
        self.shuffle = shuffle
        self.fx = open(f_x)
        self.fy = open(f_y)
        self.batch_size = batch_size

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.size / self.batch_size)
    
    def __getitem__(self, index):
        'Generate one batch of data'
#        print('Generating one batch of data of size', self.batch_size)        
        encoder_input_data = np.zeros((self.batch_size, INPUT_SIZE, INPUT_VOCAB_SIZE), dtype=np.int) 
        decoder_input_data = np.zeros((self.batch_size, OUTPUT_SIZE, OUTPUT_VOCAB_SIZE), dtype=np.float64)
        decoder_target_data = np.zeros((self.batch_size, OUTPUT_SIZE, OUTPUT_VOCAB_SIZE), dtype=np.float64)
        for j in range(self.batch_size):
            line_x, line_y = self.fx.readline(), self.fy.readline()
            question = line_x_to_indices(line_x.strip())
            expected = line_y_to_indices(line_y.strip())
            encoder_input_data[j] = ctable.encode_one_hot(question, typ="char")
            decoder_input_data[j] = ctable.encode_one_hot(expected)
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[j] = np.roll(decoder_input_data[j], -1, axis=0)

        if self.shuffle:
            indices = np.arange(self.batch_size)
            np.random.shuffle(indices)
            encoder_input_data = encoder_input_data[indices]
            decoder_input_data = decoder_input_data[indices]
            decoder_target_data = decoder_target_data[indices]
        return [encoder_input_data, decoder_input_data], decoder_target_data

    def on_epoch_end(self):
        print("End of epoch for generator of batch size", self.batch_size)
        self.fx.seek(0)
        self.fy.seek(0)


def input_generator(nsamples, train=True):
    print('Generating input for ', 'training' if train else 'validation')
    f_x, f_y = (train_x, train_y) if train else (val_x, val_y)
    with open(f_x) as fx, open(f_y) as fy:
        j = 0
        encoder_input_data = np.zeros((nsamples, INPUT_SIZE, INPUT_VOCAB_SIZE), dtype=np.int) 
        decoder_input_data = np.zeros((nsamples, OUTPUT_SIZE, OUTPUT_VOCAB_SIZE), dtype=np.float64)
        decoder_target_data = np.zeros((nsamples, OUTPUT_SIZE, OUTPUT_VOCAB_SIZE), dtype=np.float64)
        for line_x, line_y in zip(fx, fy):
            question = line_x_to_indices(line_x.strip())
            expected = line_y_to_indices(line_y.strip())
            encoder_input_data[j] = ctable.encode_one_hot(question, typ="char")
            decoder_input_data[j] = ctable.encode_one_hot(expected)
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[j] = np.roll(decoder_input_data[j], -1, axis=0)
#            print("Raw-c", line_x)
#            print("Ind-c", question)
#            print("1hot-c", encoder_input_data[j])
#            print("Raw-w", line_y)
#            print("Ind-w", expected)
#            print("1hot-w", decoder_input_data[j])
#            print("1hot-w-t", decoder_target_data[j])
            j = j + 1
            if j % nsamples == 0:
                indices = np.arange(nsamples)
                np.random.shuffle(indices)
                encoder_input_data = encoder_input_data[indices]
                decoder_input_data = decoder_input_data[indices]
                decoder_target_data = decoder_target_data[indices]

                yield [encoder_input_data, decoder_input_data], decoder_target_data
                j = 0
                encoder_input_data = np.zeros((nsamples, INPUT_SIZE, INPUT_VOCAB_SIZE), dtype=np.int) 
                decoder_input_data = np.zeros((nsamples, OUTPUT_SIZE, OUTPUT_VOCAB_SIZE), dtype=np.float64)
                decoder_target_data = np.zeros((nsamples, OUTPUT_SIZE, OUTPUT_VOCAB_SIZE), dtype=np.float64)
        print("End of ", 'training' if train else 'validation')

## Test
#t1 = list("Hello World! Foo bar, I say")
#t1i = [ctable.char_indices[c] for c in t1]
#print(t1i)
#onehot = ctable.encode_one_hot(t1i, typ="char")
#print(ctable.decode(onehot, typ="char"))
#
#t2 = ['marks', 'strongly', 'scotch', 'head', 'head', 'careless', 'animal']
#t2i = [ctable.word_indices[w] for w in t2]
#print(t2i)
#onehot = ctable.encode_one_hot(t2i)
#print(ctable.decode(onehot))

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, INPUT_VOCAB_SIZE))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, OUTPUT_VOCAB_SIZE))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(OUTPUT_VOCAB_SIZE, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

#val_gen = input_generator(1000, False)
#val_data = next(val_gen)

training_gen = DataGenerator(train_x, train_y, BATCH_SIZE, TRAINING_SIZE)
val_gen = DataGenerator(val_x, val_y, 1000, 2000)
# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
model.fit_generator(training_gen, 
        epochs=100,
        validation_data = val_gen)

# Save model
model.save('s2s.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, OUTPUT_VOCAB_SIZE))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, ctable.word_indices['@start@']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = ctable.indices_word[sampled_token_index]
        decoded_sentence.append(sampled_word)

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_word == '@end@' or len(decoded_sentence) > OUTPUT_SIZE):
           stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, OUTPUT_VOCAB_SIZE))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

val_gen_2 = input_generator(50, False)
[batch_encoder_input, batch_decoder_input], batch_decoder_target = next(val_gen_2)
for seq_index in range(50):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = batch_encoder_input[seq_index : seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', ''.join(ctable.decode(input_seq[0], typ="char")))
    print('Decoded sentence:', decoded_sentence)