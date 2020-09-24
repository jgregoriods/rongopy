import csv
import os
from rongomap import rongomap


tablets = {}

for csv_file in os.listdir('tablets'):
    with open(f'tablets/{csv_file}', 'r', encoding='utf-8-sig') as file:
        tablet = csv_file[:1]
        reader = csv.reader(file)
        head = next(reader)

        tablets[tablet] = {}

        side = head[0]
        line = 1

        for row in reader:
            tablets[tablet][f'{tablet}{side}{line}'] = row[0]
            line += 1
            if side == head[0] and line > int(head[1]):
                side = head[2]
                line = 1


def clean_data(tablets):
    clean_tablets = {}

    for tablet in tablets:
        clean_tablets[tablet] = {}

        for line in tablets[tablet]:
            new_list = []
            line_list = tablets[tablet][line].replace('.', '-').split('-')

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

            clean_tablets[tablet][line] = '-'.join(new_list)

    return clean_tablets


def simplify(tablets):
    simple_tablets = {}

    for tablet in tablets:
        simple_tablets[tablet] = {}

        for line in tablets[tablet]:
            new_list = []
            line_list = tablets[tablet][line].split('-')

            for i in line_list:
                if i in rongomap:
                    new_list.append(rongomap[i])
                else:
                    new_list.append(i)
        
            simple_tablets[tablet][line] = '-'.join(new_list)
    
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


get_stats(clean_data(tablets))
get_stats(simplify(clean_data(tablets)))