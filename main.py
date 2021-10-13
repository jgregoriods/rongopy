import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from explore.glyph_stats import GlyphStats
from explore.lang_stats import LangStats
from models.language_models import LanguageModelSVC

from data_loader import load_data


tablets = load_data('./tablets/tablets_clean.json')
corpus = load_data('./language/corpus.txt')

"""
gs = GlyphStats(tablets)

glyph_frequencies = gs.get_percentages()
print(glyph_frequencies)
print(glyph_frequencies.loc[[50]])

glyph_matrix = gs.get_matrix()
plt.imshow(glyph_matrix[:50, :50])
plt.show()

ls = LangStats(corpus)

syl_matrix = ls.get_matrix()
plt.imshow(syl_matrix)
plt.show()
"""

svc = LanguageModelSVC(corpus)
