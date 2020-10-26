import matplotlib.pyplot as plt

from rapanui import syl_matrix, syllables, corpus
from tablets import tablets_simple
from stats import get_glyph_counts, get_syl_counts
from concordance import bigram_matrix


all_syls = get_syl_counts(corpus)
syls = list(all_syls.keys())
syls.sort(key=lambda x: all_syls[x], reverse=True)
syl_freqs = [all_syls[syl] for syl in syls]

all_glyphs = get_glyph_counts(tablets_simple)
glyphs = list(all_glyphs.keys())
glyphs.sort(key=lambda x: all_glyphs[x], reverse=True)
glyph_freqs = [all_glyphs[glyph] for glyph in glyphs]


if __name__ == '__main__':
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    ax1.bar(syls, syl_freqs)

    ax2.imshow(syl_matrix)

    ax2.set_xticks(list(range(len(syllables))))
    ax2.set_xticklabels(syllables)

    ax2.set_yticks(list(range(len(syllables))))
    ax2.set_yticklabels(syllables)

    ax3.bar(glyphs[:50], glyph_freqs[:50])

    ax4.imshow(bigram_matrix)

    ax4.set_xticks(list(range(51)))
    ax4.set_xticklabels(glyphs[:50]+['999'])

    ax4.set_yticks(list(range(51)))
    ax4.set_yticklabels(glyphs[:50]+['999'])

    plt.show()