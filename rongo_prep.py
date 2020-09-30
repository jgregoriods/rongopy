import csv
import os
from rongomap import rongomap
import re


tablets = {}

for csv_file in os.listdir('tablets'):
    with open(f'tablets/{csv_file}', 'r', encoding='utf-8-sig') as file:
        tablet = csv_file[:1]
        reader = csv.reader(file)
        head = next(reader)

        tablets[tablet] = {}

        side = head[0]
        line = 1

        tablets[tablet][f'{tablet}{side}'] = {}

        for row in reader:
            tablets[tablet][f'{tablet}{side}'][f'{tablet}{side}{line}'] = row[0]
            line += 1
            if side == head[0] and line > int(head[1]):
                side = head[2]
                line = 1
                tablets[tablet][f'{tablet}{side}'] = {}


def clean_data(tablets):
    clean_tablets = {}

    for tablet in tablets:
        clean_tablets[tablet] = {}

        for side in tablets[tablet]:
            clean_tablets[tablet][side] = {}

            for line in tablets[tablet][side]:
                new_list = []
                line_list = tablets[tablet][side][line].replace('.', '-').split('-')

                for i in line_list:
                    if ':' in i:
                        a, b = i.split(':')
                        new_list.append(''.join([c for c in b if c.isdigit()]))
                        new_list.append(''.join([c for c in a if c.isdigit()]))

                    else:
                        new_list.append(''.join([c for c in i if c.isdigit()]))

                for i in range(len(new_list)):
                    for j in range(3 - len(new_list[i])):
                        new_list[i] = '0' + new_list[i]

                clean_tablets[tablet][side][line] = '-'.join(new_list)

    return clean_tablets


def simplify(tablets):
    simple_tablets = {}

    for tablet in tablets:
        simple_tablets[tablet] = {}

        for side in tablets[tablet]:
            simple_tablets[tablet][side] = {}

            for line in tablets[tablet][side]:
                new_list = []
                line_list = tablets[tablet][side][line].split('-')

                for i in line_list:
                    if i in rongomap:
                        new_list.append(rongomap[i])
                    else:
                        new_list.append(i)

                simple_tablets[tablet][side][line] = '-'.join(new_list)

    return simple_tablets


def get_glyph_counts(tablets):
    total_length = 0
    glyphs = {}
    for tablet in tablets:
        for line in tablets[tablet]:
            line_glyphs = tablets[tablet][line].split('-')
            total_length += len(line_glyphs)
            for glyph in line_glyphs:
                if glyph in glyphs:
                    glyphs[glyph] += 1
                else:
                    glyphs[glyph] = 1
    return (total_length, glyphs)


def get_stats(tablets):
    length, glyphs = get_glyph_counts(tablets)

    glyph_list = [g for g in glyphs]
    glyph_list.sort(key=lambda x: glyphs[x], reverse=True)

    sum_pct = 0
    i = 0
    for g in glyph_list:
        print(i, g, glyphs[g] / length, sum_pct)
        sum_pct += glyphs[g] / length
        i += 1

        if sum_pct >= 0.95:
            print('============================\n')
            break


def concordance(tablets):
    trigrams = {}
    previous = {}
    next_ = {}

    for tablet in tablets:
        for side in tablets[tablet]:
            for line in tablets[tablet][side]:
                glyphs = tablets[tablet][side][line].split('-')

                for i in range(len(glyphs)):
                    if i > 0 and i < len(glyphs) - 1:
                        # Trigrams
                        if glyphs[i] not in trigrams:
                            trigrams[glyphs[i]] = {}
                        a, b = glyphs[i-1], glyphs[i+1]
                        if (a, b) in trigrams[glyphs[i]]:
                            trigrams[glyphs[i]][(a, b)] += 1
                        else:
                            trigrams[glyphs[i]][(a, b)] = 1

                    # Previous
                    if i > 0:
                        if glyphs[i] not in previous:
                            previous[glyphs[i]] = {}
                        if glyphs[i-1] in previous[glyphs[i]]:
                            previous[glyphs[i]][glyphs[i-1]] += 1
                        else:
                            previous[glyphs[i]][glyphs[i-1]] = 1

                    # Next
                    if i < len(glyphs) - 1:
                        if glyphs[i] not in next_:
                            next_[glyphs[i]] = {}
                        if glyphs[i+1] in next_[glyphs[i]]:
                            next_[glyphs[i]][glyphs[i+1]] += 1
                        else:
                            next_[glyphs[i]][glyphs[i+1]] = 1

    return trigrams, previous, next_


def search_glyph(glyph, concord):
    trigrams, previous, next_ = concord

    # Trigrams
    neighbors = [k for k in list(trigrams[glyph].keys())
                 if trigrams[glyph][k] > 1 and '000' not in k]
    neighbors.sort(key=lambda x: trigrams[glyph][x], reverse=True)

    # Previous
    previous_glyphs = [p for p in list(previous[glyph].keys())
                       if previous[glyph][p] > 1 and p != '000']
    previous_glyphs.sort(key=lambda x: previous[glyph][x], reverse=True)

    # Next
    next_glyphs = [n for n in list(next_[glyph].keys())
                   if next_[glyph][n] > 1 and n != '000']
    next_glyphs.sort(key=lambda x: next_[glyph][x], reverse=True)

    print('\nTrigrams\n')
    for n in neighbors:
        freq = trigrams[glyph][n]
        print(f"   {n[0]}  |  {glyph}  |  {n[1]}  ......  x{freq}")
    print('\nBigrams\n')
    for p in previous_glyphs:
        freq_p = previous[glyph][p]
        print(f"   {p}  |  {glyph}  |       ......  x{freq_p}")
    for n_ in next_glyphs:
        freq_n = next_[glyph][n_]
        print(f"        |  {glyph}  |  {n_}  ......  x{freq_n}")

c = concordance(clean_data(tablets))
search_glyph('600', c)