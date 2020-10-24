import pandas as pd

from rapanui import syllables, corpus
from stats import get_syl_counts, get_glyph_counts
from tablets import tablets_simple
from random import shuffle, choice


def make_crypto_line(line):
    random_key = {}
    syls_cp = syllables.copy()
    shuffle(syls_cp)
    for i in range(len(syllables)):
        random_key[syllables[i]] = syls_cp[i]
    crypto_line = []
    for i in range(0, len(line), 2):
        crypto_line.append(random_key[line[i:i+2]])
    return ''.join(crypto_line)


def make_shuffled_line(line):
    shuffled_line = []
    for i in range(0, len(line), 2):
        shuffled_line.append(line[i:i+2])
    shuffle(shuffled_line)
    return ''.join(shuffled_line)


def make_random_line(line):
    random_line = []
    for i in range(0, len(line), 2):
        random_line.append(choice(syllables))
    return ''.join(random_line)


def create_pseudo_corpus(corpus):
    random_corpus = []
    crypto_corpus = []
    shuffled_corpus = []
    for verse in corpus:
        random_corpus.append(make_random_line(verse))
        crypto_corpus.append(make_crypto_line(verse))
        shuffled_corpus.append(make_shuffled_line(verse))
    pd.DataFrame(random_corpus).to_csv('random.txt', header=None, index=None)
    pd.DataFrame(crypto_corpus).to_csv('crypto.txt', header=None, index=None)
    pd.DataFrame(shuffled_corpus).to_csv('shuffled.txt', header=None, index=None)
