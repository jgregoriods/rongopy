import tensorflow as tf
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout, Bidirectional

from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from rapanui import corpus


MAX_LEN = 50  # max number of syllables per verse
NUM_SYLS = 50

real_rapanui = []
random_rapanui = []
#shuffled_rapanui = []
crypto_rapanui = []

raw_random = pd.read_csv('language/random.txt', header=None).values
random_txt = [i[0] for i in raw_random]

#raw_shuffled = pd.read_csv('language/shuffled.txt', header=None).values
#shuffled_txt = [i[0] for i in raw_shuffled]

raw_crypto = pd.read_csv('language/crypto.txt', header=None).values
crypto_txt = [i[0] for i in raw_crypto]


def separate_syllables(raw_corpus, new_list):
    for verse in raw_corpus:
        line = []
        for i in range(0, len(verse), 2):
            line.append(verse[i:i+2])
            if len(line) >= MAX_LEN:
                new_list.append([' '.join(line)])
                line = []
        new_list.append([' '.join(line)])
    return new_list


separate_syllables(corpus, real_rapanui)
separate_syllables(random_txt, random_rapanui)
#separate_syllables(shuffled_txt, shuffled_rapanui)
separate_syllables(crypto_txt, crypto_rapanui)

for verse in corpus:
    line = []
    for i in range(0, len(verse), 2):
        line.append(verse[i:i+2])
        if len(line) >= MAX_LEN:
            real_rapanui.append([' '.join(line)])
            line = []
    real_rapanui.append([' '.join(line)])


real = pd.DataFrame(real_rapanui, columns=['text'])
real['label'] = 0

rnd = pd.DataFrame(random_rapanui, columns=['text'])
rnd['label'] = 1

#shf = pd.DataFrame(shuffled_rapanui, columns=['text'])
#shf['label'] = 2

crp = pd.DataFrame(crypto_rapanui, columns=['text'])
crp['label'] = 2

all_texts = pd.concat([real, rnd, crp], ignore_index=True)
all_texts = shuffle(all_texts)
all_texts.reset_index(inplace=True)

training_size = int(len(all_texts) * 0.8)

texts = all_texts['text']
labels = all_texts['label']

X_train = texts[:training_size]
y_train = labels[:training_size]

X_test = texts[training_size:]
y_test = labels[training_size:]

#####
tfidf_vect_ngram_chars = TfidfVectorizer(ngram_range=(2,4), max_features=50)
tfidf_vect_ngram_chars.fit(texts)
xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(texts)
linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)
#####

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
    """
    model = Sequential([
        Embedding(NUM_SYLS, 32),
        Bidirectional(LSTM(64)),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train_padded, y_train, epochs=100,
              validation_data=(X_test_padded, y_test))

    model.save('lstm_model')
    """
    print('Score:', linear_svc.score(X_test, y_test))