import numpy as np
import pandas as pd


def get_glyph_counts(tablets):
    glyph_counts = {}
    for tablet in tablets:
        for line in tablets[tablet]:
            glyphs = tablets[tablet][line].split('-')
            for glyph in glyphs:
                if glyph != '000':
                    if glyph in glyph_counts:
                        glyph_counts[glyph] += 1
                    else:
                        glyph_counts[glyph] = 1
    total = sum(glyph_counts.values())
    for glyph in glyph_counts:
        glyph_counts[glyph] /= total
    return glyph_counts


def get_stats(tablets):
    glyph_counts = get_glyph_counts(tablets)
    glyph_list = [glyph for glyph in glyph_counts]
    glyph_list.sort(key=lambda x: glyph_counts[x], reverse=True)
    pct = [glyph_counts[glyph] for glyph in glyph_list]
    cum_pct = np.cumsum(pct)
    stats = pd.DataFrame(np.stack((glyph_list, pct, cum_pct), axis=1))
    stats.columns = ['Glyph', 'Pct', 'Cum Pct']
    return stats


def get_syl_counts(corpus):
    syl_counts = {}
    for verse in corpus:
        for i in range(0, len(verse) - 2, 2):
            syl = verse[i:i+2]
            if syl not in syl_counts:
                syl_counts[syl] = 0
            syl_counts[syl] += 1
    total = sum(syl_counts.values())
    for syl in syl_counts:
        syl_counts[syl] /= total
    return syl_counts
