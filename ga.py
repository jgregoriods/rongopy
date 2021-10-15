import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse

from tensorflow import keras
from time import time
from math import ceil
from random import random, randint
from tqdm import tqdm

#from tablets import tablets_simple, tablets_clean
#from rapanui import corpus
#from stats import get_glyph_counts, get_syl_counts
#from language_models import vectorizer, preprocess

from config import MAX_VERSE_LEN, SYLLABLES
from explore.lang_stats import LangStats

"""
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
"""

class Genome:
    def __init__(self, genes=None, score=None, freq=False):
        random_syllables = SYLLABLES.copy()
        if not freq:
            np.random.shuffle(random_syllables)
        self.freq = freq
        self.genes = genes or random_syllables
        self.score = score

    def mutate(self):
        j, k = np.random.choice([i for i in range(len(self.genes) - 1)],
                                size=2, replace=False)
        self.genes[j], self.genes[k] = self.genes[k], self.genes[j]
        self.score = None


class GeneticAlgorithm:
    def __init__(self, tablets, language_model, glyphs, pop_size, n_parents, n_elite,
                 prob_cross, prob_mut):
        self.tablets = tablets
        self.language_model = language_model
        self.glyphs = glyphs
        self.pop_size = pop_size
        self.n_parents = n_parents
        self.n_elite = n_elite
        self.n_children = ceil(pop_size / n_parents * 2)
        self.prob_cross = prob_cross
        self.prob_mut = prob_mut

        print('\nInitializing population')
        self.genomes = [Genome() for i in range(self.pop_size)]
        for genome in tqdm(self.genomes):
            if genome.score is None:
                key = {self.glyphs[i]: genome.genes[i] for i in range(len(SYLLABLES))}
                decoded = self.decode(key)
                genome.score = self.get_fitness(decoded)
        self.genomes.sort(key=lambda x: x.score, reverse=True)
        self.max_scores = [self.genomes[0].score]
        self.avg_scores = [np.mean([genome.score for genome in self.genomes])]
        self.best_key = {}

    def ox1(self, parent1, parent2):
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

    def decode(self, key, keep_lines=False):
        if keep_lines:
            decoded = {}
            for tablet in self.tablets:
                decoded[tablet] = {}
                for line in self.tablets[tablet]:
                    decoded_line = []
                    for glyph in self.tablets[tablet][line].split('-'):
                        if glyph in key:
                            decoded_line.append(key[glyph])
                        else:
                            decoded_line.append(glyph)
                    decoded[tablet][line] = '-'.join(decoded_line)
        else:
            decoded = []
            for tablet in self.tablets:
                for line in self.tablets[tablet]:
                    decoded_line = []
                    for glyph in self.tablets[tablet][line].split('-'):
                        if glyph in key:
                            decoded_line.append(key[glyph])
                        elif len(decoded_line) >= MAX_VERSE_LEN:
                            decoded.append(' '.join(decoded_line))
                            decoded_line = []
        return decoded

    def get_fitness(self, decoded):
        preprocessed = self.language_model.preprocess(decoded)
        probs = self.language_model.model.predict(preprocessed)
        return probs.mean(axis=0)[0]

    def evolve(self, generations):
        print('\nEvolving')
        for i in range(generations):
            print(f'\nGeneration {i+1}')
            elite = self.genomes[:self.n_elite]
            parents = self.genomes[:self.n_parents]
            np.random.shuffle(parents)
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
                    key = {self.glyphs[i]: child.genes[i] for i in range(len(SYLLABLES))}
                    decoded = self.decode(key)
                    child.score = self.get_fitness(decoded)
            self.genomes = elite + children
            self.genomes.sort(key=lambda x: x.score, reverse=True)
            self.genomes = self.genomes[:self.pop_size]
            self.max_scores.append(self.genomes[0].score)
            self.avg_scores.append(np.mean([genome.score
                                            for genome in self.genomes]))
            print(f'Best: {self.max_scores[-1]}\tAvg: {self.avg_scores[-1]}')
        self.best_key = {self.glyphs[i]: self.genomes[0].genes[i] for i in range(len(SYLLABLES))}
        plt.plot(self.max_scores)
        plt.plot(self.avg_scores)
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--popSize', type=int, default=500)
    parser.add_argument('-p', '--nParents', type=int, default=200)
    parser.add_argument('-e', '--nElite', type=int, default=50)
    parser.add_argument('-c', '--probCross', type=float, default=0.8)
    parser.add_argument('-m', '--probMut', type=float, default=0.1)
    parser.add_argument('-g', '--generations', type=int, default=200)

    args = parser.parse_args()

    ga = GeneticAlgorithm(pop_size=args.popSize, n_parents=args.nParents,
                          n_elite=args.nElite, prob_cross=args.probCross,
                          prob_mut=args.probMut)
    ga.evolve(args.generations)

    print(ga.best_key)
    with open(f'ga{int(time())}.pickle', 'wb') as file:
        pickle.dump(ga, file)


if __name__ == '__main__':
    main()
