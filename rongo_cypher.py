import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import tensorflow as tf
from collections import Counter

from lstm import preprocess


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

syls = rapa_syllables.copy()
shuffle(syls)
syl_map = {k: str(v) for k, v in zip(syls, list(range(50)))}
inverted = {str(v): k for k, v in zip(syls, list(range(50)))}

model = tf.keras.models.load_model('lstm_model')


def get_fitness(line):
    syls = []
    for i in range(0, len(line), 2):
        syls.append(line[i:i+2])
    syls = [' '.join(syls)]
    pad_syls = preprocess(syls)
    return model.predict(pad_syls)[0][0]


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
        for _ in range(50):
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
    dna_pool = [k.split('-') for k, v in sorted_dna[:25]]

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