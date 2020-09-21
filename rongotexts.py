from rongomap import rongomap
import csv


valid_chars = ['a', 'b', 'c', 'd', 'f'] + list(str(i) for i in range(10))

o = ['999', '10', '62', '62', '52', '62', '63', '999', '62', '10']
s = ['61', '6', '6', '6', '62', '10', '10', '10', '10', '62', '62', '62',
     '10', '62', '62', '62', '62', '62', '10', '10', '62', '62', '62',
     '10', '62', '10', '10']


def make_str(raw_str):
    raw_list = raw_str.replace('.', '-').split('-')
    new_list = []
    for i in range(len(raw_list)):
        if ':' in raw_list[i]:
            a, b = raw_list[i].split(':')
            raw_list[i] = b
            new_list.append(b)
            new_list.append(a)
        else:
            new_list.append(raw_list[i])

    final_list = []
    s_counter = 0
    o_counter = 0
    for i in range(len(new_list)):
        glyph = ''.join(c for c in new_list[i] if c in valid_chars)
        if len(glyph) > 3:
            if glyph[:4] in rongomap:
                final_list.append(rongomap[glyph[:4]])
            elif glyph[:3] in rongomap:
                final_list.append(rongomap[glyph[:3]])
            else:
                final_list.append('999')
        else:
            if glyph in rongomap:
                final_list.append(rongomap[glyph])
            else:
                final_list.append('999')
        if 'f' in glyph and glyph[:3] != '059' and glyph[:2] != '52':
            final_list.append('3')
        if 's' in glyph:
            final_list.append(s[s_counter])
            s_counter += 1
        if 'o' in glyph:
            final_list.append(o[o_counter])
            o_counter += 1
    return '-'.join(final_list)


pozd_enc = []
with open('tahua.csv', 'r', encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    for row in reader:
        pozd_enc.append(make_str(row[0]))
