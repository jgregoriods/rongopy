import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


rapa_syllables = ["'a", "'e", "'i", "'o", "'u",
                  'ka', 'ke', 'ki', 'ko', 'ku',
                  'ta', 'te', 'ti', 'to', 'tu',
                  'ra', 're', 'ri', 'ro', 'ru',
                  'ma', 'me', 'mi', 'mo', 'mu',
                  'na', 'ne', 'ni', 'no', 'nu',
                  'ga', 'ge', 'gi', 'go', 'gu',
                  'ha', 'he', 'hi', 'ho', 'hu',
                  'pa', 'pe', 'pi', 'po', 'pu',
                  'va', 've', 'vi', 'vo', 'vu']


def add_glottal(s):
    vowels = ['a', 'e', 'i', 'o', 'u']
    if s[0] in vowels:
        s = "'" + s
    for i in range(2):
        s = s.replace('aa', "a'a").replace('ae', "a'e").\
            replace('ai', "a'i").replace('ao', "a'o").\
            replace('au', "a'u").\
            replace('ea', "e'a").replace('ee', "e'e").\
            replace('ei', "e'i").replace('eo', "e'o").\
            replace('eu', "e'u").\
            replace('ia', "i'a").replace('ie', "i'e").\
            replace('ii', "i'i").replace('io', "i'o").\
            replace('iu', "i'u").\
            replace('oa', "o'a").replace('oe', "o'e").\
            replace('oi', "o'i").replace('oo', "o'o").\
            replace('ou', "o'u").\
            replace('ua', "u'a").replace('ue', "u'e").\
            replace('ui', "u'i").replace('uo', "u'o").\
            replace('uu', "u'u")
    return s


markov = {syl: {} for syl in rapa_syllables}
crp =''

corpus = pd.read_csv('rapanui.txt', header=None)
for line in corpus.values:
    line_f = add_glottal(line[0].replace(' ', ''))
    crp += line_f
    for i in range(0, len(line_f) - 4, 2):
        syl = line_f[i:i+2]
        nxt = line_f[i+2:i+4]
        #if syl not in markov:
        #    markov[syl] = {}
        if nxt not in markov[syl]:
            markov[syl][nxt] = 0
        markov[syl][nxt] += 1

syl_freqs = [sum(markov[syl].values()) for syl in rapa_syllables]
total_syls = sum(syl_freqs)
syl_freqs = [syl / total_syls for syl in syl_freqs]

for syl in markov:
    total = sum(markov[syl].values())
    for i in markov[syl]:
        markov[syl][i] /= total


def get_score(s):
    relative_freqs = {}
    for i in range(0, len(s) - 4, 2):
        syl = s[i:i+2]
        nxt = s[i+2:i+4]
        if syl not in relative_freqs:
            relative_freqs[syl] = {}
        if nxt not in relative_freqs[syl]:
            relative_freqs[syl][nxt] = 0
        relative_freqs[syl][nxt] += 1

    for syl in relative_freqs:
        total = sum(relative_freqs[syl].values())
        for nxt in relative_freqs[syl]:
            relative_freqs[syl][nxt] /= total

    score = 0

    for syl in relative_freqs:
        for nxt in relative_freqs[syl]:
            if relative_freqs[syl][nxt] in markov[syl]:
                score += abs(relative_freqs[syl][nxt] - markov[syl][nxt])
            else:
                score += relative_freqs[syl][nxt]

    score /= ((len(s) // 2) - 2)

    return 1 / score


M = np.zeros((50, 50))
for i in range(50):
    syl = rapa_syllables[i]
    for j in range(50):
        nxt = rapa_syllables[j]
        if nxt in markov[syl]:
            M[i, j] = markov[syl][nxt]


if __name__ == "__main__":

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax2.imshow(M)
    ax2.set_xticks(list(range(50)))
    ax2.set_xticklabels(rapa_syllables)
    ax2.set_yticks(list(range(50)))
    ax2.set_yticklabels(rapa_syllables)

    ax1.bar(list(range(50)), syl_freqs)
    ax1.set_xticks(list(range(50)))
    ax1.set_xticklabels(rapa_syllables)

    plt.show()

    snt = ''
    for i in range(15):
        snt += np.random.choice(rapa_syllables)
    
    k = np.random.randint(0, len(crp) - 30)
    if crp[k] in ['a', 'e', 'i', 'o', 'u']:
        k += 1
    print(crp[k:k+30])
    print(get_score(crp[k:k+30]))
    print(snt)
    print(get_score(snt))