import tensorflow as tf
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.utils import shuffle

from old.rapanui import corpus


MIN_LEN = 10  # min number of syllables per verse
MAX_LEN = 50  # max number of syllables per verse

raw_random = pd.read_csv('language/random.txt', header=None).values
random_txt = [i[0] for i in raw_random]

raw_crypto = pd.read_csv('language/crypto.txt', header=None).values
crypto_txt = [i[0] for i in raw_crypto]


def separate_syllables(raw_corpus):
    separated = []
    for verse in raw_corpus:
        line = []
        for i in range(0, len(verse), 2):
            line.append(verse[i:i+2])
            if len(line) >= MAX_LEN:
                separated.append([' '.join(line)])
                line = []
        if len(line) >= MIN_LEN:
            separated.append([' '.join(line)])
    return separated


real_rapanui = separate_syllables(corpus)
random_rapanui = separate_syllables(random_txt)
crypto_rapanui = separate_syllables(crypto_txt)

real = pd.DataFrame(real_rapanui, columns=['text'])
real['label'] = 0

rnd = pd.DataFrame(random_rapanui, columns=['text'])
rnd['label'] = 1

crp = pd.DataFrame(crypto_rapanui, columns=['text'])
crp['label'] = 2

#data = pd.concat([real, rnd, crp], ignore_index=True)
data = pd.concat([real, crp], ignore_index=True)
data = shuffle(data)
data.reset_index(inplace=True)

texts = data['text']
labels = data['label']
