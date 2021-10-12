import json
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

from config import MAX_GLYPHS


class GlyphModel:
    def __init__(self, raw_data):
        self.tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
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

    def make_training_data(self):
        sequences = []
        for text in self.encoded_data:
            for i in range(1, MAX_GLYPHS+1):
                sequences.append(text[:i+1])
            for i in range(1, len(text)-MAX_GLYPHS):
                sequences.append(text[i:i+MAX_GLYPHS+1])
        #self.max_sequence_len = max([len(sequence) for sequence in sequences])

        sequences = np.array(pad_sequences(sequences, maxlen=MAX_GLYPHS+1, padding='pre'))

        X = sequences[:, :-1]
        y = to_categorical(sequences[:, -1], num_classes=self.vocab_size)

        return X, y

    def build(self, embed_size, lstm_size):
        self.model = Sequential([
            Embedding(self.vocab_size, embed_size, input_length=MAX_GLYPHS),
            LSTM(lstm_size),
            Dropout(0.5),
            Dense(self.vocab_size, activation='softmax')
        ])

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X, y, validation_split, epochs):
        self.history = self.model.fit(X, y, validation_split=validation_split, epochs=epochs, verbose=1)
        plt.plot(self.history.history['val_accuracy'])
        plt.show()


if __name__ == '__main__':
    tm = GlyphModel()
    tm.load_data('./tablets/tablets_clean.json')
    X, y = tm.make_training_data()
    tm.build(32, 128)
    tm.train(X, y, 0.33, 20)
