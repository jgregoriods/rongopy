import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout,\
                                    Bidirectional

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.utils import shuffle

from config import SYLLABLES, MIN_VERSE_LEN, MAX_VERSE_LEN


class CorpusLabeller:
    def __init__(self, corpus):
        self.corpus = corpus
        self.crypto_corpus = self.encrypt_corpus()
        self.shuffled_corpus = self.shuffle_corpus()
        self.random_corpus = self.randomize_corpus()
        self.labelled_texts = self.label_texts()

    def encrypt_line(self, line):
        random_key = {}
        syls_cp = SYLLABLES.copy()
        np.random.shuffle(syls_cp)
        for i in range(len(SYLLABLES)):
            random_key[SYLLABLES[i]] = syls_cp[i]
        crypto_line = []
        for i in range(0, len(line), 2):
            crypto_line.append(random_key[line[i:i+2]])
        return ''.join(crypto_line)

    def shuffle_line(self, line):
        shuffled_line = []
        for i in range(0, len(line), 2):
            shuffled_line.append(line[i:i+2])
        np.random.shuffle(shuffled_line)
        return ''.join(shuffled_line)

    def randomize_line(self, line):
        random_line = []
        for i in range(0, len(line), 2):
            random_line.append(np.random.choice(SYLLABLES))
        return ''.join(random_line)

    def encrypt_corpus(self):
        crypto_corpus = []
        for verse in self.corpus:
            crypto_corpus.append(self.encrypt_line(verse))
        return crypto_corpus

    def randomize_corpus(self):
        random_corpus = []
        for verse in self.corpus:
            random_corpus.append(self.randomize_line(verse))
        return random_corpus

    def shuffle_corpus(self):
        shuffled_corpus = []
        for verse in self.corpus:
            shuffled_corpus.append(self.shuffle_line(verse))
        return shuffled_corpus

    def truncate(self, text):
        separated = []
        for verse in text:
            line = []
            for i in range(0, len(verse), 2):
                line.append(verse[i:i+2])
                if len(line) >= MAX_VERSE_LEN:
                    separated.append([' '.join(line)])
                    line = []
            if len(line) >= MIN_VERSE_LEN:
                separated.append([' '.join(line)])
        return separated

    def label_texts(self):
        real_corpus = self.truncate(self.corpus)
        real_corpus_df = pd.DataFrame(real_corpus, columns=['text'])
        real_corpus_df['label'] = 0

        pseudo_corpus = self.truncate(self.crypto_corpus)
        pseudo_corpus_df = pd.DataFrame(pseudo_corpus, columns=['text'])
        pseudo_corpus_df['label'] = 1

        all_texts = pd.concat([real_corpus_df, pseudo_corpus_df], ignore_index=True)
        all_texts = shuffle(all_texts)
        all_texts.reset_index(inplace=True)

        return all_texts
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


class LanguageModelSVC:
    def __init__(self, labelled_texts):
        self.classifier = CalibratedClassifierCV(LinearSVC())
        self.texts = labelled_texts['text']
        self.labels = labelled_texts['label']
        self.vectorizer = TfidfVectorizer(ngram_range=(2, 6))

    def make_training_data(self, test_split):
        self.vectorizer.fit(self.texts)
        ngrams = self.vectorizer.transform(self.texts)

        train_size = int(len(self.texts) * (1 - test_split))
        X_train = ngrams[:train_size]
        X_test = ngrams[train_size:]

        y_train = self.labels[:train_size]
        y_test = self.labels[train_size:]

        return X_train, y_train, X_test, y_test

    def train(self, X_train, y_train, X_test=None, y_test=None):
        self.classifier.fit(X_train, y_train)
        if X_test is not None and y_test is not None:
            print('LinearSVC score:', self.classifier.score(X_test, y_test))

    def predict(self, x):
        probs = self.classifier.predict_proba(self.vectorizer.transform(x))
        return probs
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


class LanguageModelLSTM:
    def __init__(self, labelled_texts):
        self.tokenizer = Tokenizer()
        self.texts = labelled_texts['text']
        self.labels = labelled_texts['label']
        self.vocab_size = 0
        self.model = None

    def preprocess(self, text):
        tokenized = self.tokenizer.texts_to_sequences(text)
        padded = pad_sequences(tokenized, maxlen=MAX_VERSE_LEN,
                               padding='post', truncating='post')
        return padded

    def make_training_data(self, test_split):
        train_size = int(len(self.texts) * (1 - test_split))

        X_train_raw = self.texts[:train_size]
        X_test_raw = self.texts[train_size:]

        y_train = self.labels[:train_size]
        y_test = self.labels[train_size:]

        self.tokenizer.fit_on_texts(self.texts)
        self.vocab_size = len(self.tokenizer.word_index) + 1

        X_train = self.preprocess(X_train_raw)
        X_test = self.preprocess(X_test_raw)

        return X_train, y_train, X_test, y_test

    def build(self, embedding_size, lstm_size, dropout):
        self.model = Sequential([
            Embedding(self.vocab_size, embedding_size),
            Bidirectional(LSTM(lstm_size)),
            Dropout(dropout),
            Dense(2, activation='softmax')
        ])

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X, y, validation_split, epochs):
        self.model.fit(X, y, validation_split=validation_split, epochs=epochs, verbose=1)

    def save(self, filename):
        self.model.save(f'saved_models/lstm/{filename}')
        self.model = None
        with open(f'saved_models/{filename}.pickle', 'wb') as f:
            pickle.dump(self, f)
        self.model = load_model(f'saved_models/lstm/{filename}')