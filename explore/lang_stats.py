import numpy as np
import pandas as pd
import pickle

from config import SYLLABLES, VOWELS


class LangStats:
    def __init__(self, raw_corpus):
        self.corpus = self.preprocess_corpus(raw_corpus)

    def get_syllable_counts(self):
        syllable_counts = {}
        for verse in self.corpus:
            for i in range(0, len(verse) - 2, 2):
                syl = verse[i:i+2]
                if syl not in syllable_counts:
                    syllable_counts[syl] = 0
                syllable_counts[syl] += 1
        total = sum(syllable_counts.values())
        for syl in syllable_counts:
            syllable_counts[syl] /= total
        return syllable_counts

    def glottalize(self, string):
        string = string.replace(' ', '')
        if string[0] in VOWELS:
            string = "'" + string
        return (''.join(i + ("'" if i in VOWELS and j in VOWELS else '')
                        for i, j in zip(string[:-1], string[1:]))) + string[-1]

    def preprocess_corpus(self, raw_corpus):
        corpus = []
        for verse in raw_corpus:
            corpus.append(self.glottalize(verse[0]))
        return corpus

    def get_matrix(self):
        n = len(SYLLABLES)
        transition_matrix = np.zeros((n, n))
        for line in self.corpus:
            for i in range(0, len(line) - 2, 2):
                syl = line[i:i+2]
                nxt = line[i+2:i+4]
                j = SYLLABLES.index(syl)
                k = SYLLABLES.index(nxt)
                transition_matrix[j, k] += 1
        row_total = transition_matrix.sum(axis=1)
        for i in range(len(transition_matrix)):
            if row_total[i]:
                transition_matrix[i] /= row_total[i]
        return transition_matrix

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
