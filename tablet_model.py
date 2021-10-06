import json
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense


with open('./tablets/tablets_clean.json') as file:
    tablets = json.load(file)

data = []
for tablet in tablets:
    for line in tablets[tablet]:
        data += tablets[tablet][line].split('-')
data_str = ' '.join(data)

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data_str])

VOCAB_SIZE = len(tokenizer.word_index) + 1  # 570

encoded_data = []
for tablet in tablets:
    for line in tablets[tablet]:
        encoded_data.append(tokenizer.texts_to_sequences([tablets[tablet][line]])[0])

sequences = []
for text in encoded_data:
    for i in range(1, len(text)):
        sequences.append(text[:i+1])
MAX_LEN = max([len(sequence) for sequence in sequences])

sequences = np.array(pad_sequences(sequences, maxlen=MAX_LEN, padding='pre'))

X = sequences[:, :-1]
y = to_categorical(sequences[:, -1], num_classes=VOCAB_SIZE)

model = Sequential()
model.add(Embedding(VOCAB_SIZE, 32, input_length=MAX_LEN - 1))
model.add(LSTM(64))
model.add(Dense(VOCAB_SIZE, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=100, verbose=2)
