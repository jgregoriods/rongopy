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
    def __init__(self, genes=None, score=None):
        self.genes = genes or syls.copy()
        self.score = score or self.get_score()

    def get_score(self):
        key = {glyphs[i]: self.genes[i] for i in range(len(syls))}
        decoded = decode_tablets(tablets_simple, key)
        #self.score = get_fitness(decoded)
        return get_fitness(decoded)

    def mutate(self):
        # new_genes = self.genes.copy()
        i = randint(0, len(syls) - 2)
        self.genes[i], self.genes[i+1] = self.genes[i+1], self.genes[i]
        self.score = self.get_score()
        #new_genes[i], new_genes[i+1] = new_genes[i+1], new_genes[i]
        #new_score = self.get_score(new_genes)
        #if new_score > self.score:
        #    self.genes = new_genes
        #    self.score = new_score


class GeneticAlgorithm:
    def __init__(self, pop_size, n_parents, n_children, prob_mut):
        self.pop_size = pop_size
        self.prob_mut = prob_mut
        self.n_parents = n_parents
        self.n_children = n_children
        self.genomes = [Genome() for i in range(self.pop_size)]
        # self.get_scores(self.genomes)
        self.genomes.sort(key=lambda x: x.score, reverse=True)
        self.max_scores = [self.genomes[0].score]
        self.avg_scores = [np.mean([genome.score for genome in self.genomes])]
        self.best_key = {}

    # def get_scores(self, genomes):
    #     print('Calculating scores')
    #     for genome in tqdm(genomes):
    #         genome.get_score()

    def evolve(self, generations):
        print('\nEvolving')
        for i in range(generations):
            print(f'\n================== Generation {i+1} ==================')
            parents = self.genomes[:self.n_parents]
            children = []
            for parent in tqdm(parents):
                for i in range(self.n_children):
                    child = Genome(parent.genes, parent.score)
                    if random() < self.prob_mut:
                        child.mutate()
                    children.append(child)
            self.genomes = parents + children
            for i in tqdm(range(self.pop_size - (len(self.genomes)))):
                self.genomes.append(Genome())
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
    ga = GeneticAlgorithm(pop_size=1000, n_parents=200, n_children=2,
                          prob_mut=0.2)
    ga.evolve(50)
    print(ga.best_key)
    pickle.dump(ga, open('ga.pickle', 'wb'))
