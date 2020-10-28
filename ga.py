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
        rnd_syls = syls.copy()
        shuffle(rnd_syls)
        self.genes = genes or rnd_syls
        self.score = score or self.get_score()

    def get_score(self):
        key = {glyphs[i]: self.genes[i] for i in range(len(syls))}
        decoded = decode_tablets(tablets_simple, key)
        return get_fitness(decoded)

    def mutate(self):
        for i in range(np.random.poisson(0.5) + 1):
            j = np.random.poisson(0.5) + 1
            k = randint(0, len(self.genes) - (j+1))
            self.genes[k], self.genes[k+j] = self.genes[k+j], self.genes[k]
        self.score = self.get_score()


class GeneticAlgorithm:
    def __init__(self, pop_size, n_parents, n_elite, prob_cross, prob_mut):
        self.pop_size = pop_size
        self.n_parents = n_parents
        self.n_elite = n_elite
        self.n_children = pop_size // n_parents * 2
        self.prob_cross = prob_cross
        self.prob_mut = prob_mut

        self.genomes = [Genome() for i in tqdm(range(self.pop_size))]
        # self.genomes = [Genome()]
        # self.std_score = self.genomes[0].score
        # self.genomes += [Genome(score=self.genomes[0].score)
        #                  for i in range(self.pop_size - 1)]
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
            print(f'\n================= Generation {i+1} =================')
            elite = self.genomes[:self.n_elite]
            parents = self.genomes[:self.n_parents]
            shuffle(parents)
            children = []
            for i in tqdm(range(0, len(parents), 2)):
                parent1 = parents[i]
                parent2 = parents[i+1]
                for i in range(self.n_children):
                    if random() < self.prob_cross:
                        new_genes = self.erx(parent1.genes, parent2.genes)
                        child = Genome(new_genes)
                        children.append(child)
                    else:
                        child = Genome(parent1.genes, parent1.score)
                        if random() < self.prob_mut:
                            child.mutate()
                        children.append(child)
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
                          prob_cross=0.8, prob_mut=0.2)
    ga.evolve(100)
    print(ga.best_key)
    pickle.dump(ga, open('ga.pickle', 'wb'))
