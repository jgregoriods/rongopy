import tensorflow as tf
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout, Bidirectional
from sklearn.utils import shuffle

from rapanui import corpus


MAX_LEN = 50  # max number of syllables per verse
NUM_SYLS = 50

real_rapanui = []
fake_rapanui = []

#raw_shuffled = pd.read_csv('language/shuffled.txt', header=None).values
#shuffled = [i[0] for i in raw_shuffled]

raw_random = pd.read_csv('language/random.txt', header=None).values
random_txt = [i[0] for i in raw_random]

for verse in corpus:
    line = []
    for i in range(0, len(verse), 2):
        line.append(verse[i:i+2])
        if len(line) >= MAX_LEN:
            real_rapanui.append([' '.join(line)])
            line = []
    if len(line) >= 10:
        real_rapanui.append([' '.join(line)])

for verse in random_txt:
    line = []
    for i in range(0, len(verse), 2):
        line.append(verse[i:i+2])
        if len(line) >= MAX_LEN:
            fake_rapanui.append([' '.join(line)])
            line = []
    if len(line) >= 10:
        fake_rapanui.append([' '.join(line)])


real = pd.DataFrame(real_rapanui, columns=['text'])
real['label'] = 0

fake = pd.DataFrame(fake_rapanui, columns=['text'])
fake['label'] = 1

all_texts = pd.concat([real, fake], ignore_index=True)
all_texts = shuffle(all_texts)
all_texts.reset_index(inplace=True)

training_size = int(len(all_texts) * 0.8)

texts = all_texts['text']
labels = all_texts['label']

X_train = texts[:training_size]
y_train = labels[:training_size]

X_test = texts[training_size:]
y_test = labels[training_size:]

tokenizer = Tokenizer(num_words=NUM_SYLS, oov_token='<OOV>')
tokenizer.fit_on_texts(real['text'])

X_train_seqs = tokenizer.texts_to_sequences(X_train)
X_test_seqs = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_seqs, maxlen=MAX_LEN,
                               padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seqs, maxlen=MAX_LEN,
                              padding='post', truncating='post')


def preprocess(line):
    line_seqs = tokenizer.texts_to_sequences(line)
    line_padded = pad_sequences(line_seqs, maxlen=MAX_LEN,
                                padding='post', truncating='post')
    return line_padded

"""
model = Sequential()

model.add(Embedding(NUM_SYLS, 32))
model.add(LSTM(32, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(2, activation='softmax'))
"""

if __name__ == '__main__':
    model = Sequential([
        Embedding(NUM_SYLS, 32),
        Bidirectional(LSTM(64)),
        Dropout(0.2),
        Dense(2, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train_padded, y_train, epochs=5,
              validation_data=(X_test_padded, y_test))

    model.save('lstm_model')
