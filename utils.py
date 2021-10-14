import numpy as np
import pandas as pd

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
