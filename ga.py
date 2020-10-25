import keras
import pickle
import numpy as np
import matplotlib.pyplot as plt

from random import shuffle, random, randint
from tqdm import tqdm

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


class Genome:
    def __init__(self, genes=None):
        self.genes = genes or syls.copy()
        self.score = 0

    def get_score(self):
        key = {glyphs[i]: self.genes[i] for i in range(len(syls))}
        decoded = decode_tablets(tablets_simple, key)
        self.score = get_fitness(decoded)

    def mutate(self):
        i = randint(0, len(syls) - 2)
        self.genes[i], self.genes[i+1] = self.genes[i+1], self.genes[i]


class GeneticAlgorithm:
    def __init__(self, pop_size, generations, prob_mut, n_parents):
        self.pop_size = pop_size
        self.generations = generations
        self.prob_mut = prob_mut
        self.n_parents = n_parents
        self.genomes = [Genome() for i in range(self.pop_size)]
        self.get_scores(self.genomes)
        self.genomes.sort(key=lambda x: x.score, reverse=True)
        self.max_scores = [self.genomes[0].score]
        self.avg_scores = [np.mean([genome.score for genome in self.genomes])]
        self.best_key = {}

    def get_scores(self, genomes):
        print('Calculating scores')
        for genome in tqdm(genomes):
            genome.get_score()

    def evolve(self):
        print('\nEvolving')
        for i in range(self.generations):
            print(f'\n================== Generation {i+1} ==================')
            parents = self.genomes[:self.n_parents]
            children = []
            while len(children) < self.pop_size - self.n_parents:
                for parent in parents:
                    child = Genome(parent.genes)
                    if random() < self.prob_mut:
                        child.mutate()
                    children.append(child)
            self.get_scores(children)
            self.genomes = parents + children
            self.genomes.sort(key=lambda x: x.score, reverse=True)
            self.max_scores.append(self.genomes[0].score)
            self.avg_scores.append(np.mean([genome.score
                                            for genome in self.genomes]))
            print(f'Best: {self.max_scores[-1]}\tAvg: {self.avg_scores[-1]}')
        self.best_key = {glyphs[i]: self.genomes[0].genes[i]
                         for i in range(len(syls))}
        plt.plot(self.max_scores)
        plt.plot(self.avg_scores)
        plt.show()


if __name__ == '__main__':
    ga = GeneticAlgorithm(pop_size=10, generations=1, prob_mut=0.2,
                          n_parents=1)
    ga.evolve()
    print(ga.best_key)
    pickle.dump(ga, open('ga.pickle', 'wb'))
