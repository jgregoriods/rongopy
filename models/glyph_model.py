import json
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

from config import MAX_GLYPHS


class GlyphModel:
    def __init__(self, raw_data):
        self.tokenizer = Tokenizer(50, oov_token='<OOV>')
        self.vocab_size = 0
        self.raw_data = raw_data
        self.encoded_data = self.preprocess_data()
        #self.max_sequence_len = 0
        self.model = None
        self.history = None

    def preprocess_data(self):
        data = []
        for tablet in self.raw_data:
            for line in self.raw_data[tablet]:
                data += self.raw_data[tablet][line].split('-')
        data_str = ' '.join(data)

        self.tokenizer.fit_on_texts([data_str])
        self.vocab_size = len(self.tokenizer.word_index) + 1

        encoded_data = []
        for tablet in self.raw_data:
            for line in self.raw_data[tablet]:
                encoded_data.append(self.tokenizer.texts_to_sequences([self.raw_data[tablet][line]])[0])

        return encoded_data

    def make_training_data(self, test_split):
        sequences = []
        for text in self.encoded_data:
            for i in range(1, MAX_GLYPHS+1):
                sequences.append(text[:i+1])
            for i in range(1, len(text)-MAX_GLYPHS):
                sequences.append(text[i:i+MAX_GLYPHS+1])
        #self.max_sequence_len = max([len(sequence) for sequence in sequences])

        sequences = np.array(pad_sequences(sequences, maxlen=MAX_GLYPHS+1, padding='pre'))
        np.random.shuffle(sequences)

        X = sequences[:, :-1]
        y = to_categorical(sequences[:, -1], num_classes=self.vocab_size)

        train_size = int(len(sequences) * (1 - test_split))
        X_train = X[:train_size]
        X_test = X[train_size:]

        y_train = y[:train_size]
        y_test = y[train_size:]

        return X_train, y_train, X_test, y_test

    def build(self, embedding_size, lstm_size, dropout):
        self.model = Sequential([
            Embedding(self.vocab_size, embedding_size, input_length=MAX_GLYPHS),
            LSTM(lstm_size),
            Dropout(dropout),
            Dense(self.vocab_size, activation='softmax')
        ])

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X, y, validation_split, batch_size, epochs):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
        self.history = self.model.fit(X, y, validation_split=validation_split, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[es])
        plt.plot(self.history.history['val_accuracy'])
        plt.show()


if __name__ == '__main__':
    tm = GlyphModel()
    tm.load_data('./tablets/tablets_clean.json')
    X, y = tm.make_training_data()
    tm.build(32, 128)
    tm.train(X, y, 0.33, 20)
