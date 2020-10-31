import keras
import pickle
import numpy as np
import matplotlib.pyplot as plt

from time import time
from math import ceil
from random import shuffle, random, randint
from tqdm import tqdm

from tablets import tablets_simple, tablets_clean
from rapanui import corpus, syllables
from stats import get_glyph_counts, get_syl_counts
from language_models import vectorizer, preprocess


svc = pickle.load(open('models/svc_model', 'rb'))
lstm = keras.models.load_model('models/lstm_model')

selected = ['A', 'B', 'C', 'D', 'E', 'G', 'N', 'P', 'R', 'S']
tablets_subset = {k: tablets_simple[k] for k in selected}

glyph_dict = get_glyph_counts(tablets_subset)
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
    def __init__(self, genes=None, score=None, freq=False):
        rnd_syls = syls.copy()
        if not freq:
            shuffle(rnd_syls)
        self.freq = freq
        self.genes = genes or rnd_syls
        self.score = score

    def get_score(self):
        key = {glyphs[i]: self.genes[i] for i in range(len(syls))}
        decoded = decode_tablets(tablets_subset, key)
        self.score = get_fitness(decoded)

    def mutate(self):
        j, k = np.random.choice([i for i in range(len(self.genes) - 1)],
                                size=2, replace=False)
        self.genes[j], self.genes[k] = self.genes[k], self.genes[j]
        self.score = None


class GeneticAlgorithm:
    def __init__(self, pop_size, n_parents, n_elite, prob_cross, prob_mut):
        self.pop_size = pop_size
        self.n_parents = n_parents
        self.n_elite = n_elite
        self.n_children = ceil(pop_size / n_parents * 2)
        self.prob_cross = prob_cross
        self.prob_mut = prob_mut

        print('\nInitializing population')
        self.genomes = [Genome(freq=True) for i in range(self.pop_size)]
        for genome in tqdm(self.genomes):
            if genome.score is None:
                genome.get_score()
        self.genomes.sort(key=lambda x: x.score, reverse=True)
        self.max_scores = [self.genomes[0].score]
        self.avg_scores = [np.mean([genome.score for genome in self.genomes])]
        self.best_key = {}

    def orx(self, parent1, parent2):
        i = randint(0, len(parent1) - 1)
        j = randint(i + 1, len(parent1))
        segment = parent1[i:j]
        missing = [k for k in parent2 if k not in segment]
        pref, suff = missing[:i], missing[i:]
        return pref + segment + suff

    def erx(self, parent1, parent2):
        nodes = {gene: set() for gene in parent1}
        for i in range(len(parent1) - 1):
            nodes[parent1[i]].add(parent1[i+1])
            nodes[parent2[i]].add(parent2[i+1])
        child = [np.random.choice([parent1[0], parent2[0]])]
        for node in nodes:
            if child[-1] in nodes[node]:
                nodes[node].remove(child[-1])
        while(len(child) < len(parent1)):
            if nodes[child[-1]]:
                next_node = np.random.choice(list(nodes[child[-1]]))
            else:
                next_node = np.random.choice([k for k in nodes.keys()
                                              if k not in child])
            for node in nodes:
                if next_node in nodes[node]:
                    nodes[node].remove(next_node)
            child.append(next_node)
        return child

    def evolve(self, generations):
        print('\nEvolving')
        for i in range(generations):
            print(f'\nGeneration {i+1}')
            elite = self.genomes[:self.n_elite]
            parents = self.genomes[:self.n_parents]
            shuffle(parents)
            children = []
            for i in range(0, len(parents), 2):
                parent1 = parents[i]
                parent2 = parents[i+1]
                for i in range(self.n_children):
                    if random() < self.prob_cross:
                        new_genes = self.erx(parent1.genes, parent2.genes)
                        child = Genome(new_genes)
                    else:
                        child = Genome(parent1.genes, parent1.score)
                    if random() < self.prob_mut:
                        child.mutate()
                    children.append(child)
            for child in tqdm(children):
                if child.score is None:
                    child.get_score()
            self.genomes = elite + children
            self.genomes.sort(key=lambda x: x.score, reverse=True)
            self.genomes = self.genomes[:self.pop_size]
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
    ga = GeneticAlgorithm(pop_size=500, n_parents=200, n_elite=50,
                          prob_cross=0.8, prob_mut=0.1)
    ga.evolve(20)
    print(ga.best_key)
    with open(f'ga{int(time())}.pickle', 'wb') as file:
        pickle.dump(ga, file)
