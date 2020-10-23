import tensorflow as tf
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout,\
                                    Bidirectional

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from text_prepare import texts, labels, training_size, NUM_SYLS, MAX_LEN,\
                         tokenizer


X_train = texts[:training_size]
y_train = labels[:training_size]

X_test = texts[training_size:]
y_test = labels[training_size:]

vectorizer = TfidfVectorizer(ngram_range=(2, 4))
vectorizer.fit(texts)
ngrams = vectorizer.transform(texts)

X_train_seqs = tokenizer.texts_to_sequences(X_train)
X_test_seqs = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_seqs, maxlen=MAX_LEN,
                               padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seqs, maxlen=MAX_LEN,
                              padding='post', truncating='post')


if __name__ == '__main__':
    linear_svc = LinearSVC()
    linear_svc.fit(ngrams[:training_size], y_train)

    pickle.dump(linear_svc, open('svc_model', 'wb'))

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