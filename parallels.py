import pandas as pd
import re

from gru.utils import clean_char
from horley_encoding import horley_encoding


horley_pars = pd.read_csv("horley_parallels.csv").values[:,2]

simplified_parallels = []

for seq in horley_pars:
    seq = seq.split("#")[0]
    glyphs = seq.replace("-", " ").split()
    clean = ' '.join([clean_char(glyph) for glyph in glyphs])
    simplified = []
    for glyph in clean.split():
        if glyph in horley_encoding:
            simplified.append(horley_encoding[glyph])
        else:
            num = re.sub("[a-z]", "", glyph)
            if num in horley_encoding:
                simplified.append(horley_encoding[num])
            else:
                simplified.append(glyph)
    simplified_parallels.append(' '.join(simplified))
simplified_parallels = [i.split() for i in simplified_parallels]

parallels_no_200 = []
for par in simplified_parallels:
    parallels_no_200.append([i for i in par if i != '200'])
