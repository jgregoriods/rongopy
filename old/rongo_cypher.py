import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from tensorflow import keras
from collections import Counter

from bigrams import rapa_syllables
from lstm import preprocess
#from fitness import get_fitness

"""
rapa_words = []

vowels = ['a', 'e', 'i', 'o', 'u']
with open('rapanui.txt') as file:
    for line in file.readlines():
        for word in line.split(' '):
            if word[0] in vowels:
                word = "'" + word
            word = word.replace('\n', '').\
                        replace('aa', "a'a").replace('ae', "a'e").\
                        replace('ai', "a'i").replace('ao', "a'o").\
                        replace('au', "a'u").\
                        replace('ea', "e'a").replace('ee', "e'e").\
                        replace('ei', "e'i").replace('eo', "e'o").\
                        replace('eu', "e'u").\
                        replace('ia', "i'a").replace('ie', "i'e").\
                        replace('ii', "i'i").replace('io', "i'o").\
                        replace('iu', "i'u").\
                        replace('oa', "o'a").replace('oe', "o'e").\
                        replace('oi', "o'i").replace('oo', "o'o").\
                        replace('ou', "o'u").\
                        replace('ua', "u'a").replace('ue', "u'e").\
                        replace('ui', "u'i").replace('uo', "u'o").\
                        replace('uu', "u'u")
            rapa_words.append(word)

word_list = set(rapa_words)
rapa_dict = Counter(rapa_words)

def splitString(s):
    found = []

    def rec(stringLeft, wordsSoFar):
        if not stringLeft:
            found.append(wordsSoFar)
        for pos in range(1, len(stringLeft)+1):
            if stringLeft[:pos] in word_list:
                rec(stringLeft[pos:], wordsSoFar + [stringLeft[:pos]])

    rec(s.lower(), [])
    if found:
        found.sort(key=lambda x: len(x))
        return ' '.join(found[0])
    elif len(s) > 6:
        spl = len(s) // 2
        if s[spl] in vowels:
            spl += 1
        part1 = splitString(s[:spl])
        part2 = splitString(s[spl:])
        return part1 + ' ' + part2
    else:
        return s

trainDF = pd.read_csv('rn3.csv')
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(trainDF['text'])

#train_x, test_x, train_y, test_y = train_test_split(xtrain_tfidf_ngram_chars, trainDF['Label'], test_size=0.05)
X, Y = xtrain_tfidf_ngram_chars, trainDF['Label']

linear_svc = LinearSVC()
clf = CalibratedClassifierCV(linear_svc)
clf.fit(X, Y)

#stc = "'ekoro'ekorehanu'i'e"
#stc_t = tfidf_vect_ngram_chars.transform(pd.Series(stc))


corpus = []

with open('rapanui.txt') as file:
    corpus = file.read()

for i in range(2):
    corpus = corpus.replace(' ', '').replace('\n', '').\
                    replace('aa', "a'a").replace('ae', "a'e").\
                    replace('ai', "a'i").replace('ao', "a'o").\
                    replace('au', "a'u").\
                    replace('ea', "e'a").replace('ee', "e'e").\
                    replace('ei', "e'i").replace('eo', "e'o").\
                    replace('eu', "e'u").\
                    replace('ia', "i'a").replace('ie', "i'e").\
                    replace('ii', "i'i").replace('io', "i'o").\
                    replace('iu', "i'u").\
                    replace('oa', "o'a").replace('oe', "o'e").\
                    replace('oi', "o'i").replace('oo', "o'o").\
                    replace('ou', "o'u").\
                    replace('ua', "u'a").replace('ue', "u'e").\
                    replace('ui', "u'i").replace('uo', "u'o").\
                    replace('uu', "u'u")

rapa_syllables = ["'a", "'e", "'i", "'o", "'u",
                  'ka', 'ke', 'ki', 'ko', 'ku',
                  'ta', 'te', 'ti', 'to', 'tu',
                  'ra', 're', 'ri', 'ro', 'ru',
                  'ma', 'me', 'mi', 'mo', 'mu',
                  'na', 'ne', 'ni', 'no', 'nu',
                  'ga', 'ge', 'gi', 'go', 'gu',
                  'ha', 'he', 'hi', 'ho', 'hu',
                  'pa', 'pe', 'pi', 'po', 'pu',
                  'va', 've', 'vi', 'vo', 'vu']

syl_indices = {k: v for k, v in zip(rapa_syllables, list(range(48)))}

# Markov matrix
M = np.ones((48, 48))

# Initial state
pi = np.zeros(48)


# Update Markov matrix
def update_transition(syl1, syl2):
    i = syl_indices[syl1]
    j = syl_indices[syl2]
    M[i, j] += 1


# Update initial state
def update_pi(syl):
    i = syl_indices[syl]
    pi[i] += 1


for i in range(0, len(corpus) - 2, 2):
    syl1 = corpus[i:i+2]
    syl2 = corpus[i+2:i+4]

    update_pi(syl1)
    update_transition(syl1, syl2)


# Normalise probabilities
pi /= pi.sum()
M /= M.sum(axis=1, keepdims=True)


def get_seq_prob(seq):
    logp = 0
    for i in range(0, len(seq) - 2, 2):
        syl1 = syl_indices[seq[i:i+2]]
        syl2 = syl_indices[seq[i+2:i+4]]
        logp += np.log(M[syl1, syl2]) + np.log(pi[syl1])
    return logp

"""
syls = rapa_syllables.copy()
shuffle(syls)
syl_map = {k: str(v) for k, v in zip(syls, list(range(50)))}
inverted = {str(v): k for k, v in zip(syls, list(range(50)))}

model = keras.load_model('../lstm_model')

def get_fitness(line):
    syls = []
    for i in range(0, len(line), 2):
        syls.append(line[i:i+2])
    syls = [' '.join(syls)]
    return model.predict(syls)[0]


original = "'a'ure'a'ohovehikihaho'ena'ohovehinu'ina'otito'okumatu'a'ero'amarego'eka'itagatamohatu'o'o'u'e'ure'e"
print(get_fitness(original))

def encode(text, key):
    encoded = []
    for i in range(0, len(text), 2):
        encoded.append(syl_map[text[i:i+2]])
    return '-'.join(encoded)


def decode(text, key):
    decoded = []
    lst = text.split('-')
    for i in lst:
        decoded.append(key[i])
    return ''.join(decoded)


# Genetic Algorithm ####################################


dna_pool = []
for _ in range(100):
    dna = rapa_syllables.copy()
    shuffle(dna)
    dna_pool.append(dna)


def evolve_offspring(dna_pool, n_children):
    offspring = []

    for dna in dna_pool:
        for _ in range(n_children):
            copy = dna.copy()
            j = np.random.randint(len(copy))
            k = np.random.randint(len(copy))
            # Swap
            copy[j], copy[k] = copy[k], copy[j]
            offspring.append(copy)

    return offspring + dna_pool


num_iters = 1000
scores = np.zeros(num_iters)
best_dna = None
best_map = None
best_score = float('-inf')

encoded = encode(original, syl_map)
print(encoded)

for i in range(num_iters):
    if i > 0:
        # Get offspring
        dna_pool = evolve_offspring(dna_pool, 2)
        for _ in range(50000):
            dna = rapa_syllables.copy()
            shuffle(dna)
            dna_pool.append(dna)

    dna2score = {}

    for dna in dna_pool:
        current_map = {str(k): v for k, v in zip(list(range(50)), dna)}
        decoded = decode(encoded, current_map)

        score = get_fitness(decoded)

        dna2score['-'.join(dna)] = score

        if score > best_score:
            best_dna = dna
            best_map = current_map
            best_score = score

    scores[i] = np.mean(list(dna2score.values()))

    sorted_dna = sorted(dna2score.items(), key=lambda x: x[1], reverse=True)
    dna_pool = [k.split('-') for k, v in sorted_dna[:25000]]

    if i % 100 == 0:
        print('iter:', i, 'score:', scores[i], 'best so far:', best_score)
        print(decode(encoded, best_map))

plt.plot(scores)
plt.show()
dcd = decode(encoded, best_map)
for i in range(len(dcd)):
    if dcd[i] == original[i]:
        print(dcd[i], end='')
    else:
        print('*', end='')
print('')