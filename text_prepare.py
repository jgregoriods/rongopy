import tensorflow as tf
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.utils import shuffle

from rapanui import corpus


MAX_LEN = 50   # max number of syllables per verse
NUM_SYLS = 50  # total number of syllables in the language

real_rapanui = []
random_rapanui = []
crypto_rapanui = []

raw_random = pd.read_csv('language/random.txt', header=None).values
random_txt = [i[0] for i in raw_random]

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


def preprocess(line):
    line_seqs = tokenizer.texts_to_sequences(line)
    line_padded = pad_sequences(line_seqs, maxlen=MAX_LEN,
                                padding='post', truncating='post')
    return line_padded


separate_syllables(corpus, real_rapanui)
separate_syllables(random_txt, random_rapanui)
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

crp = pd.DataFrame(crypto_rapanui, columns=['text'])
crp['label'] = 2

all_texts = pd.concat([real, rnd, crp], ignore_index=True)
all_texts = shuffle(all_texts)
all_texts.reset_index(inplace=True)

training_size = int(len(all_texts) * 0.9)

texts = all_texts['text']
labels = all_texts['label']

tokenizer = Tokenizer(num_words=NUM_SYLS, oov_token='<OOV>')
tokenizer.fit_on_texts(real['text'])