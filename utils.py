from rapanui import syllables, corpus
from stats import get_syl_counts, get_glyph_counts
from tablets import tablets_simple


all_syls = get_syl_counts(corpus)
syls = list(all_syls.keys())
syls.sort(key=lambda x: all_syls[x], reverse=True)

all_glyphs = get_glyph_counts(tablets_simple)
glyphs = list(all_glyphs.keys())
glyphs.sort(key=lambda x: all_glyphs[x], reverse=True)


key = {glyphs[i]: syls[i] for i in range(len(syls))}

def decode(line, key):
    decoded = []
    glyphs = line.split('-')
    for glyph in glyphs:
        if glyph in key:
            decoded.append(key[glyph])
        else:
            decoded.append(glyph)
    return '-'.join(decoded)

