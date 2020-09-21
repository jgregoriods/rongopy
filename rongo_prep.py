import csv
import os

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
    valid_chars = [str(i) for i in range(10)]

    clean_tablets = {}

    for tablet in tablets:
        clean_tablets[tablet] = {}

        for line in tablets[tablet]:
            new_list = []
            line_list = tablets[tablet][line].replace('.', '-').split('-')

            for i in line_list:
                if ':' in i:
                    a, b = i.split(':')
                    new_list.append(''.join([c for c in a if c.isdigit()]))
                    new_list.append(''.join([c for c in b if c.isdigit()]))

                else:
                    new_list.append(''.join([c for c in i if c.isdigit()]))

            for i in range(len(new_list)):
                for j in range(3 - len(new_list[i])):
                    new_list[i] = '0' + new_list[i]

            clean_tablets[tablet][line] = '-'.join(new_list)

    return clean_tablets


total_len = 0


def get_stats(tablets):
    global total_len
    glyphs = {}
    for tablet in tablets:
        for line in tablets[tablet]:
            line_glyphs = tablets[tablet][line].split('-')
            total_len += len(line_glyphs)
            for glyph in line_glyphs:
                if glyph in glyphs:
                    glyphs[glyph] += 1
                else:
                    glyphs[glyph] = 1
    return glyphs


stats = get_stats(clean_data(tablets))
l = [g for g in stats]
l.sort(key=lambda x: stats[x], reverse=True)

sum_percent = 0
i = 0
for g in l:
    print(i, g, sum_percent)
    sum_percent += stats[g] / total_len
    i += 1
    if sum_percent >= 0.95:
        print('=======================')
        break

print(clean_data(tablets))