import pandas as pd
import numpy as np


corpus = []
raw_corpus = pd.read_csv('language/corpus.txt', header=None).values

syllables = ["'a", "'e", "'i", "'o", "'u",
             'ka', 'ke', 'ki', 'ko', 'ku',
             'ta', 'te', 'ti', 'to', 'tu',
             'ra', 're', 'ri', 'ro', 'ru',
             'ma', 'me', 'mi', 'mo', 'mu',
             'na', 'ne', 'ni', 'no', 'nu',
             'ga', 'ge', 'gi', 'go', 'gu',
             'ha', 'he', 'hi', 'ho', 'hu',
             'pa', 'pe', 'pi', 'po', 'pu',
             'va', 've', 'vi', 'vo', 'vu']


def glottalize(s):
    vowels = 'aeiou'

    s = s.replace(' ', '')

    if s[0] in vowels:
        s = "'" + s

    return (''.join(x + ("'" if x in vowels and nxt in vowels else '')
                    for x, nxt in zip(s[:-1], s[1:]))) + s[-1]


for verse in raw_corpus:
    corpus.append(glottalize(verse[0]))

syl_matrix = np.zeros((50, 50))

for line in corpus:
    for i in range(0, len(line) - 2, 2):
        syl = line[i:i+2]
        nxt = line[i+2:i+4]
        if syl not in syllables or nxt not in syllables:
            print(line)
        j = syllables.index(syl)
        k = syllables.index(nxt)
        syl_matrix[j, k] += 1

syl_sums = syl_matrix.sum(axis=1)

for i in range(len(syl_matrix)):
    if syl_sums[i] > 0:
        syl_matrix[i] /= syl_sums[i]
