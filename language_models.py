import tensorflow as tf
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout,\
                                    Bidirectional

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer

from text_prepare import real, texts, labels, MAX_LEN


NUM_SYLS = 50  # total number of syllables in the language
TRAIN_SIZE = int(len(texts) * 0.9)

X_train = texts[:TRAIN_SIZE]
y_train = labels[:TRAIN_SIZE]

X_test = texts[TRAIN_SIZE:]
y_test = labels[TRAIN_SIZE:]

vectorizer = TfidfVectorizer(ngram_range=(2, 3))
vectorizer.fit(texts)

ngrams = vectorizer.transform(texts)
ngrams_train = ngrams[:TRAIN_SIZE]
ngrams_test = ngrams[TRAIN_SIZE:]

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


if __name__ == '__main__':
    # SVC model
    linear_svc = LinearSVC()
    clf = CalibratedClassifierCV(linear_svc)
    clf.fit(ngrams_train, y_train)

    print('LinearSVC score:', clf.score(ngrams_test, y_test))

    pickle.dump(clf, open('svc_model', 'wb'))

    # LSTM model
    lstm_model = Sequential([
        Embedding(NUM_SYLS, 32),
        Bidirectional(LSTM(64)),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])

    lstm_model.compile(loss='sparse_categorical_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])

    lstm_model.fit(X_train_padded, y_train, epochs=30,
                   validation_data=(X_test_padded, y_test))

    lstm_model.save('lstm_model')