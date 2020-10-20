import os
import pandas as pd

from glyphmap import glyphmap, appendages, feathered


def clean_line(line, keep_adorn=False):
    clean = []
    glyph_list = line.replace('.', '-').split('-')
    for glyph in glyph_list:
        # Glyphs on top of each other, marked by ':' in the CEIPP
        # notation, should generally be read bottom to top. Thus, their
        # order should be inverted in the list.
        if ':' in glyph:
            stacked_glyphs = glyph.split(':')
            for stacked_glyph in reversed(stacked_glyphs):
                cleaned_glyph = ''.join([c for c in stacked_glyph
                                         if c.isdigit()]).zfill(3)
                if keep_adorn:
                    if 'o' in stacked_glyph:
                        cleaned_glyph += 'o'
                    if 's' in stacked_glyph:
                        cleaned_glyph += 's'
                    if 'f' in stacked_glyph and cleaned_glyph not in feathered:
                        cleaned_glyph += 'f'
                clean.append(cleaned_glyph)
        else:
            cleaned_glyph = ''.join([c for c in glyph if c.isdigit()]).zfill(3)
            if keep_adorn:
                if 'o' in glyph:
                    cleaned_glyph += 'o'
                if 's' in glyph:
                    cleaned_glyph += 's'
                if 'f' in glyph and cleaned_glyph not in feathered:
                    cleaned_glyph += 'f'
            clean.append(cleaned_glyph)
    return '-'.join(clean)


def simplify(line):
    simple = []
    for glyph in clean_line(line, keep_adorn=True).split('-'):
        if glyph[:3] in glyphmap:
            simple.append(glyphmap[glyph[:3]])
        else:
            simple.append(glyph[:3])
        for letter in 'osf':
            if letter in glyph:
                simple.append(letter)
    return simple


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

for tablet in tablets_simple:
    o_counter = 0
    s_counter = 0
    for line in tablets_simple[tablet]:
        if tablet in appendages:  # remove this later
            for i in range(len(tablets_simple[tablet][line])):
                glyph = tablets_simple[tablet][line][i]
                if glyph == 'o':
                    tablets_simple[tablet][line][i] = appendages[tablet]['o'][o_counter]
                    o_counter += 1
                elif glyph == 's':
                    tablets_simple[tablet][line][i] = appendages[tablet]['s'][s_counter]
                    s_counter += 1
                elif glyph == 'f':
                    tablets_simple[tablet][line][i] = '003'
        tablets_simple[tablet][line] = '-'.join(tablets_simple[tablet][line])
