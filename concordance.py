import numpy as np

from tablets import tablets, tablets_clean, tablets_simple
from stats import get_glyph_counts


def get_concordance(tablets):
    concordance = {}
    for tablet in tablets:
        for line in tablets[tablet]:
            glyph_list = tablets[tablet][line].split('-')
            for i in range(1, len(glyph_list) - 1):
                glyph = glyph_list[i]
                prev = glyph_list[i-1]
                next_ = glyph_list[i+1]
                if glyph not in concordance:
                    concordance[glyph] = {}
                if (prev, next_) not in concordance[glyph]:
                    concordance[glyph][(prev, next_)] = 0
                concordance[glyph][(prev, next_)] += 1
    for glyph in concordance:
        total = sum(concordance[glyph].values())
        for k in concordance[glyph]:
            concordance[glyph][k] /= total
    return concordance


concordance = get_concordance(tablets)
concordance_clean = get_concordance(tablets_clean)
concordance_simple = get_concordance(tablets_simple)

bigram_matrix = np.zeros((51, 51))

glyph_counts = get_glyph_counts(tablets_simple)
glyphs = list(glyph_counts.keys())
glyphs.sort(key=lambda x: glyph_counts[x], reverse=True)
glyphs = glyphs[:50]

for tablet in tablets_simple:
    for line in tablets_simple[tablet]:
        glyph_list = tablets_simple[tablet][line].split('-')
        for i in range(len(glyph_list) - 1):
            glyph = glyph_list[i]
            next_ = glyph_list[i+1]
            j = glyphs.index(glyph) if glyph in glyphs else 50
            k = glyphs.index(next_) if next_ in glyphs else 50
            bigram_matrix[j, k] += 1

glyph_sums = bigram_matrix.sum(axis=1)
bigram_matrix /= glyph_sums[:, np.newaxis]
