import string
import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Dense


samples = ['The cat sat on the mat.', 'The dog ate my homework.', 'pie in the sky', 'Activating Conda Environments in Powershell is not supported']

characters = string.printable
token_index = dict(zip(characters, range(1, len(characters) + 1)))
max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
        for j, character in enumerate(sample[:max_length]):
                index = token_index.get(character)
                print(f'{character} - {index}')
                results[i, j, index] = 1.


model = Sequential()
model.add(LSTM(len(characters)*max_length, input_shape = (4, 25,), activation="sigmoid"))
model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])
