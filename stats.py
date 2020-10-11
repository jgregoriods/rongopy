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
    return glyph_counts


def get_stats(tablets):
    glyph_counts = get_glyph_counts(tablets)
    total_length = sum(glyph_counts.values())
    glyph_list = [glyph for glyph in glyph_counts]
    glyph_list.sort(key=lambda x: glyph_counts[x], reverse=True)
    pct = [glyph_counts[glyph] / total_length for glyph in glyph_list]
    cum_pct = np.cumsum(pct)
    stats = pd.DataFrame(np.stack((glyph_list, pct, cum_pct), axis=1))
    stats.columns = ['Glyph', 'Pct', 'Cum Pct']
    return stats


def search_glyph(glyph, concord):
    trigrams, previous, next_ = concord

    # Trigrams
    neighbors = [k for k in list(trigrams[glyph].keys())
                 if trigrams[glyph][k] > 1 and '000' not in k]
    neighbors.sort(key=lambda x: trigrams[glyph][x], reverse=True)

    # Previous
    previous_glyphs = [p for p in list(previous[glyph].keys())
                       if previous[glyph][p] > 1 and p != '000']
    previous_glyphs.sort(key=lambda x: previous[glyph][x], reverse=True)

    # Next
    next_glyphs = [n for n in list(next_[glyph].keys())
                   if next_[glyph][n] > 1 and n != '000']
    next_glyphs.sort(key=lambda x: next_[glyph][x], reverse=True)

    print('\nTrigrams\n')
    for n in neighbors:
        freq = trigrams[glyph][n]
        print(f"   {n[0]}  |  {glyph}  |  {n[1]}  ......  x{freq}")
    print('\nBigrams\n')
    for p in previous_glyphs:
        freq_p = previous[glyph][p]
        print(f"   {p}  |  {glyph}  |       ......  x{freq_p}")
    for n_ in next_glyphs:
        freq_n = next_[glyph][n_]
        print(f"        |  {glyph}  |  {n_}  ......  x{freq_n}")