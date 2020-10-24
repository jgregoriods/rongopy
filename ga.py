import keras
import pickle
import numpy as np

from random import shuffle

from tablets import tablets_simple
from rapanui import syllables
from stats import get_glyph_counts
from language_models import vectorizer, preprocess


svc = pickle.load(open('svc_model', 'rb'))
lstm = keras.models.load_model('lstm_model')

glyph_dict = get_glyph_counts(tablets_simple)
glyphs = list(glyph_dict.keys())
glyphs.sort(key=lambda x: glyph_dict[x], reverse=True)
glyphs = glyphs[:50]

shuffle(syllables)

key = {glyphs[i]: syllables[i] for i in range(50)}


def decode_tablets(tablets, key):
    decoded = []
    for tablet in tablets:
        for line in tablets[tablet]:
            decoded_line = []
            for glyph in tablets[tablet][line].split('-'):
                if glyph in key:
                    decoded_line.append(key[glyph])
                elif len(decoded_line) >= 10:
                    decoded.append(' '.join(decoded_line))
                    decoded_line = []
    return decoded


def get_fitness(decoded):
    svc_probs = svc.predict_proba(vectorizer.transform(decoded))
    svc_score = svc_probs.mean(axis=0)[0]

    lstm_probs = lstm.predict(preprocess(decoded))
    lstm_score = lstm_probs.mean(axis=0)[0]

    return np.mean([svc_score, lstm_score])