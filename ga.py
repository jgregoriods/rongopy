import keras
import pickle
import numpy as np

from random import shuffle, randint

from tablets import tablets_simple
from rapanui import corpus, syllables
from stats import get_glyph_counts, get_syl_counts
from language_models import vectorizer, preprocess


svc = pickle.load(open('svc_model', 'rb'))
lstm = keras.models.load_model('lstm_model')

glyph_dict = get_glyph_counts(tablets_simple)
glyphs = list(glyph_dict.keys())
glyphs.sort(key=lambda x: glyph_dict[x], reverse=True)
glyphs = glyphs[:50]

syl_dict = get_syl_counts(corpus)
syls = list(syl_dict.keys())
syls.sort(key=lambda x: syl_dict[x], reverse=True)

key = {glyphs[i]: syls[i] for i in range(len(syls))}


def decode_tablets(tablets, key):
    decoded = []
    for tablet in tablets:
        for line in tablets[tablet]:
            decoded_line = []
            for glyph in tablets[tablet][line].split('-'):
                if glyph in key:
                    decoded_line.append(key[glyph])
                elif len(decoded_line) >= 10:
                    decoded.append(' '.join(decoded_line))
                    decoded_line = []
    return decoded


def get_fitness(decoded):
    svc_probs = svc.predict_proba(vectorizer.transform(decoded))
    svc_score = svc_probs.mean(axis=0)[0]

    lstm_probs = lstm.predict(preprocess(decoded))
    lstm_score = lstm_probs.mean(axis=0)[0]

    return np.mean([svc_score, lstm_score])


class GeneticAlgorithm:
    def __init__(self, pop_size):
        self.genomes = [syls.copy() for i in range(pop_size)]
        self.scores = [0 for i in range(pop_size)]

    def mutate(self):
        for genome in self.genomes:
            i = randint(0, len(syls) - 2)
            genome[i], genome[i+1] = genome[i+1], genome[i]

    def get_scores(self):
        for i in range(len(self.genomes)):
            genome = self.genomes[i]
            key = {glyphs[k]: genome[k] for k in range(len(syls))}
            decoded = decode_tablets(tablets_simple, key)
            score = get_fitness(decoded)
            self.scores[i] = score
