import json
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


class TabletModel:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.raw_data = None
        self.encoded_data = None
        self.vocab_size = 0
        self.max_sequence_len = 0
        self.model = None
        self.history = None

    def load_data(self, filepath):
        with open(filepath) as file:
            self.raw_data = json.load(file)

        data = []
        for tablet in self.raw_data:
            for line in self.raw_data[tablet]:
                data += self.raw_data[tablet][line].split('-')
        data_str = ' '.join(data)

        self.tokenizer.fit_on_texts([data_str])
        self.vocab_size = len(self.tokenizer.word_index) + 1

        self.encoded_data = []
        for tablet in self.raw_data:
            for line in self.raw_data[tablet]:
                self.encoded_data.append(self.tokenizer.texts_to_sequences([self.raw_data[tablet][line]])[0])

    def make_training_data(self):
        sequences = []
        for text in self.encoded_data:
            for i in range(1, len(text)):
                sequences.append(text[:i+1])
        self.max_sequence_len = max([len(sequence) for sequence in sequences])

        sequences = np.array(pad_sequences(sequences, maxlen=self.max_sequence_len, padding='pre'))

        X = sequences[:, :-1]
        y = to_categorical(sequences[:, -1], num_classes=self.vocab_size)

        return X, y

    def build(self, embed_size, lstm_size):
        self.model = Sequential([
            Embedding(self.vocab_size, embed_size, input_length=self.max_sequence_len - 1),
            LSTM(lstm_size),
            Dense(self.vocab_size, activation='softmax')
        ])

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X, y, validation_split, epochs):
        self.history = self.model.fit(X, y, validation_split=validation_split, epochs=epochs, verbose=2)
        plt.plot(self.history.history['val_accuracy'])
        plt.show()


if __name__ == '__main__':
    tm = TabletModel()
    tm.load_data('./tablets/tablets_clean.json')
    X, y = tm.make_training_data()
    tm.build(64, 256)
    tm.train(X, y, 0.25, 100)
