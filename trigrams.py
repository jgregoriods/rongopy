import pandas as pd
import numpy as np
from bigrams import add_glottal, crp, rapa_syllables


trigrams = {}

corpus = pd.read_csv('rapanui.txt', header=None)
for line in corpus.values:
    line_f = add_glottal(line[0].replace(' ', ''))
    for i in range(0, len(line_f) - 6, 2):
        two_syl = line_f[i:i+4]
        nxt = line_f[i+4:i+6]
        if two_syl not in trigrams:
            trigrams[two_syl] = {}
        if nxt not in trigrams[two_syl]:
            trigrams[two_syl][nxt] = 0
        trigrams[two_syl][nxt] += 1

for two_syl in trigrams:
    total = sum(trigrams[two_syl].values())
    for nxt in trigrams[two_syl]:
        trigrams[two_syl][nxt] /= total


def get_score(s):
    rel_freq = {}
    score = 0

    for i in range(0, len(s) - 4, 2):
        syls = s[i:i+4]
        nxt = s[i+4:i+6]
        if syls not in rel_freq:
            rel_freq[syls] = {}
        if nxt not in rel_freq[syls]:
            rel_freq[syls][nxt] = 0
        rel_freq[syls][nxt] += 1

    for syl in rel_freq:
        total = sum(rel_freq[syl].values())
        for nxt in rel_freq[syl]:
            rel_freq[syl][nxt] /= total

    for syls in trigrams:
        if syls in rel_freq:
            for nxt in trigrams[syls]:
                if nxt in rel_freq[syls]:
                    score += abs(trigrams[syls][nxt] - rel_freq[syls][nxt])
                else:
                    score += trigrams[syls][nxt]
        else:
            score += 1

    for syls in rel_freq:
        if syls not in trigrams:
            score += 1
        else:
            for nxt in rel_freq[syls]:
                if nxt not in trigrams[syls]:
                    score += rel_freq[syls][nxt]

    return (len(trigrams) - score) / (len(s) // 2)


if __name__ == "__main__":
    k = np.random.randint(0, len(crp) - 10)
    if crp[k] in ['a', 'e', 'i', 'o', 'u']:
        k += 1
    snt = ''
    for i in range(5):
        snt += np.random.choice(rapa_syllables)
    print(crp[k:k+10])
    print(get_score(crp[k:k+10]))
    print(snt)
    print(get_score(snt))