import csv
import numpy as np
from random import shuffle


rapa_list = []
with open('rapanui.txt', 'r') as file:
    for row in file.readlines():
        rapa_list.append(row)
rapanui = ''.join(rapa_list)

for i in range(2):
    rapanui = rapanui.replace(' ', '').replace('\n', '').\
                    replace('aa', "a'a").replace('ae', "a'e").\
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

"""
syllables = []
for i in range(0, len(rapanui), 2):
    syllables.append(rapanui[i:i+2])
syllables = '-'.join(syllables)
print(syllables)
"""
def write_corpus():
    samples = []
    for i in range(0, len(rapanui), 200):
        samples.append([rapanui[i:i+200]])

    with open('rn_corpus.csv', 'w') as file:
        writer = csv.writer(file)
        for row in samples:
            writer.writerow(row)

def make_shuffled_line(line):
    syls = []
    for i in range(0, len(line), 2):
        syls.append(line[i:i+2])
    new_syls = np.random.choice(syls, len(line) // 2, replace=False)
    return ''.join(new_syls)

def write_shuffled():
    shuffled = []
    for i in range(0, len(rapanui), 200):
        shuffled.append([make_shuffled_line(rapanui[i:i+200])])
    
    with open('rn_shuffled.csv', 'w') as file:
        writer = csv.writer(file)
        for row in shuffled:
            writer.writerow(row)

rapa_syllables = ['ki', "'a", 'ta', 'ri', 'ra', "'u", 'ku', 'ma', "'i", 'te',
                  "'e", 'hu', 'ka', 'pu', 'va', 'po', 'me', "'o", 'ho', 'ke',
                  'ro', 'ru', 'ga', 're', 'ko', 'tu', 'hi', 'mi', 'nu', 'ti',
                  'pi', 'mo', 've', 'to', 'ha', 'na', 'he', 'gi', 'ge', 'ne',
                  'pa', 'ni', 'no', 'pe', 'vo', 'go', 'mu', 'vi']

rs1 = rapa_syllables.copy()
rs2 = rapa_syllables.copy()
shuffle(rs1)
shuffle(rs2)

rnd_dic = {a: b for a, b in zip(rs1, rs2)}

def make_repl_line(line):
    syls = []
    for i in range(0, len(line), 2):
        syls.append(rnd_dic[line[i:i+2]])
    return ''.join(syls)

def write_repl():
    replaced = []
    for i in range(0, len(rapanui), 200):
        replaced.append([make_repl_line(rapanui[i:i+200])])
    
    with open('rn_replaced.csv', 'w') as file:
        writer = csv.writer(file)
        for row in replaced:
            writer.writerow(row)

def write_rnd():
    rnd = []
    for j in range(30):
        line = ''
        for i in range(100):
            line += np.random.choice(rapa_syllables)
        rnd.append([line])

    with open('rn_rnd.csv', 'w') as file:
        writer = csv.writer(file)
        for row in rnd:
            writer.writerow(row)
