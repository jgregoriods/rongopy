import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from config import MAX_VERSE_LEN, SYLLABLES


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
        self.n_children = int(np.ceil(pop_size / n_parents * 2))
        self.prob_cross = prob_cross
        self.prob_mut = prob_mut

        print('\nInitializing population...')
        self.genomes = [Genome() for i in range(self.pop_size)]
        for genome in self.genomes:
            if genome.score is None:
                key = {self.glyphs[i]: genome.genes[i] for i in range(len(SYLLABLES))}
                decoded = self.decode(key)
                genome.score = self.get_fitness(decoded)

        self.genomes.sort(key=lambda x: x.score, reverse=True)
        self.max_scores = [self.genomes[0].score]
        self.avg_scores = [np.mean([genome.score for genome in self.genomes])]
        self.best_key = {}

    def ox1(self, parent1, parent2):
        i = np.random.randint(0, len(parent1) - 1)
        j = np.random.randint(i + 1, len(parent1))
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
        print('Evolving...')
        for i in range(generations):
            elite = self.genomes[:self.n_elite]
            parents = self.genomes[:self.n_parents]
            np.random.shuffle(parents)
            children = []
            for j in range(0, len(parents), 2):
                parent1 = parents[j]
                parent2 = parents[j+1]
                for k in range(self.n_children):
                    if np.random.random() < self.prob_cross:
                        new_genes = self.erx(parent1.genes, parent2.genes)
                        child = Genome(new_genes)
                    else:
                        child = Genome(parent1.genes, parent1.score)
                    if np.random.random() < self.prob_mut:
                        child.mutate()
                    children.append(child)
            for child in children:
                if child.score is None:
                    key = {self.glyphs[x]: child.genes[x] for x in range(len(SYLLABLES))}
                    decoded = self.decode(key)
                    child.score = self.get_fitness(decoded)
            self.genomes = elite + children
            self.genomes.sort(key=lambda x: x.score, reverse=True)
            self.genomes = self.genomes[:self.pop_size]
            self.max_scores.append(self.genomes[0].score)
            self.avg_scores.append(np.mean([genome.score
                                            for genome in self.genomes]))
            print(f'\rGeneration {i+1}\tBest: {self.max_scores[-1]}\tAvg: {self.avg_scores[-1]}', end='')
        self.best_key = {self.glyphs[i]: self.genomes[0].genes[i] for i in range(len(SYLLABLES))}
        print('\n')
        plt.plot(self.max_scores)
        plt.plot(self.avg_scores)
        plt.show()
