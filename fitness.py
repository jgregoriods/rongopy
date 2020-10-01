import pandas as pd
import numpy as np

from bigrams import rapa_syllables, add_glottal


trigrams = {}
bigrams = {}

corpus = pd.read_csv('rapanui.txt', header=None)

for line in corpus.values:
    line_f = add_glottal(line[0].replace(' ', ''))
    
    for i in range(0, len(line_f) - 4, 2):
        syl = line_f[i:i+6]
        if syl not in trigrams:
            trigrams[syl] = 0
        trigrams[syl] += 1

    for i in range(0, len(line_f) - 2, 2):
        syl = line_f[i:i+4]
        if syl not in bigrams:
            bigrams[syl] = 0
        bigrams[syl] += 1

sum_tri = sum(trigrams.values())
sum_bi = sum(bigrams.values())

for t in trigrams:
    trigrams[t] /= sum_tri

for b in bigrams:
    bigrams[b] /= sum_bi


def get_score(s):
    p_tri = {}
    p_bi = {}


    for i in range(0, len(s) - 4, 2):
        syl = s[i:i+6]
        if syl not in p_tri:
            p_tri[syl] = 0
        p_tri[syl] += 1
    
    sum_p_tri = sum(p_tri.values())

    for t in p_tri:
        p_tri[t] /= sum_p_tri
    
    tri_delta = 0

    for t in trigrams:
        if t in p_tri:
            tri_delta += abs(trigrams[t] - p_tri[t])
        else:
            tri_delta += trigrams[t]

    for t in p_tri:
        if t not in trigrams:
            tri_delta += p_tri[t]

    
    for i in range(0, len(s) - 2, 2):
            syl = s[i:i+4]
            if syl not in p_bi:
                p_bi[syl] = 0
            p_bi[syl] += 1
        
    sum_p_bi = sum(p_bi.values())

    for b in p_bi:
        p_bi[b] /= sum_p_bi
    
    bi_delta = 0

    for b in bigrams:
        if b in p_bi:
            bi_delta += abs(bigrams[b] - p_bi[b])
        else:
            bi_delta += bigrams[b]

    for b in p_bi:
        if b not in bigrams:
            bi_delta += p_bi[b]
        
    return tri_delta + bi_delta


print(get_score("'etimote'ako'ako"))

snt = ''
for i in range(5):
    snt += np.random.choice(rapa_syllables)
print(snt)
print(get_score(snt))