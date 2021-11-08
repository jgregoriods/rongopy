import matplotlib.pyplot as plt
from config import TABLET_SUBSET

from explore.glyph_stats import GlyphStats
from explore.lang_stats import LangStats
from models.language_models import CorpusLabeller, LanguageModelSVC, LanguageModelLSTM
from ga.ga import GeneticAlgorithm

from utils import load_data


all_tablets = load_data('./tablets/tablets_clean.json')
tablets = {tablet: all_tablets[tablet] for tablet in TABLET_SUBSET}

raw_corpus = load_data('./language/corpus.txt')

gs = GlyphStats(tablets)

glyph_frequencies = gs.get_percentages()
print(glyph_frequencies)
print(glyph_frequencies.loc[[50]])

glyph_matrix = gs.get_matrix()
#plt.imshow(glyph_matrix[:50, :50])
#plt.show()

top_glyphs = gs.get_top_n(50)

ls = LangStats(raw_corpus)
corpus = ls.corpus

syl_matrix = ls.get_matrix()
#plt.imshow(syl_matrix)
#plt.show()

cl = CorpusLabeller(corpus)
labelled_texts = cl.labelled_texts

svc = LanguageModelSVC(labelled_texts)
X_train, y_train, X_test, y_test = svc.make_training_data(0.1)
svc.train(X_train, y_train, X_test, y_test)

lstm = LanguageModelLSTM(labelled_texts)
X_train, y_train, X_test, y_test = lstm.make_training_data(0.1)
lstm.build(32, 128, 0.2)
lstm.train(X_train, y_train, 0.1, 10)

ga = GeneticAlgorithm(tablets, lstm, top_glyphs, 500, 200, 50, 0.8, 0.1)
ga.evolve(100, 10)
ga.plot()
ga.save('ga')
