import os
import pandas as pd

from glyphmap import glyphmap


def clean_line(line):
    clean = []
    glyph_list = line.replace('.', '-').split('-')
    for glyph in glyph_list:
        # Glyphs on top of each other, marked by ':' in the CEIPP
        # notation, should generally be read bottom to top. Thus, their
        # order should be inverted in the list.
        if ':' in glyph:
            top_glyph, bottom_glyph = glyph.split(':')
            clean.append(''.join([c for c in bottom_glyph
                                  if c.isdigit()]).zfill(3))
            clean.append(''.join([c for c in top_glyph
                                  if c.isdigit()]).zfill(3))
        else:
            clean.append(''.join([c for c in glyph if c.isdigit()]).zfill(3))
    return '-'.join(clean)


def simplify(line):
    simple = []
    for glyph in clean_line(line).split('-'):
        if glyph in glyphmap:
            simple.append(glyphmap[glyph])
        else:
            simple.append(glyph)
    return '-'.join(simple)


tablets = {}
tablets_clean = {}
tablets_simple = {}


for csv_file in os.listdir('tablets'):
    file = pd.read_csv(f'tablets/{csv_file}', header=None)
    tablet = csv_file[:1]
    labels = file[0].values
    lines = file[1].values
    tablets[tablet] = {}
    tablets_clean[tablet] = {}
    tablets_simple[tablet] = {}
    for label, line in zip(labels, lines):
        tablets[tablet][label] = line
        tablets_clean[tablet][label] = clean_line(line)
        tablets_simple[tablet][label] = simplify(line)
