import pickle
import os
import json

from ga import Genome, GeneticAlgorithm, decode_tablets
from tablets import tablets_simple, tablets_clean
from stats import get_glyph_counts


selected = ['A', 'B', 'C', 'D', 'E', 'G', 'N', 'P', 'R', 'S']
tablets_subset = {k: tablets_simple[k] for k in selected}

glyph_dict = get_glyph_counts(tablets_subset)
glyphs = list(glyph_dict.keys())
glyphs.sort(key=lambda x: glyph_dict[x], reverse=True)
glyphs = glyphs[:50]

best_keys = {glyphs[i]: {} for i in range(49)}

if __name__ == '__main__':
    for filename in os.listdir('./results/simple'):
        with open(f'./results/simple/{filename}', 'rb') as f:
            ga = pickle.load(f)
        genomes = []
        for i in range(len(ga.genomes)):
            if len(genomes) >= 10:
                break
            elif ga.genomes[i] not in genomes:
                genomes.append(ga.genomes[i])
        for genome in genomes:
            key = {glyphs[i]: genome.genes[i] for i in range(49)}
            for glyph in key:
                syl = key[glyph]
                if syl in best_keys[glyph]:
                    best_keys[glyph][syl] += 1
                else:
                    best_keys[glyph][syl] = 1

        decoded = {'score': ga.max_scores[-1],
                   'text': decode_tablets(tablets_simple, ga.best_key,
                                          keep_lines=True)}
        with open(f'decoded{filename[2:-7]}.json', 'w') as file:
            json.dump(decoded, file)

    with open('keys.json', 'w') as file:
        json.dump(best_keys, file)
