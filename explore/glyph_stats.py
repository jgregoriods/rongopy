import numpy as np
import pandas as pd


class GlyphStats:
    def __init__(self, tablets):
        self.tablets = tablets

    def get_counts(self):
        glyph_counts = {}
        for tablet in self.tablets:
            for line in self.tablets[tablet]:
                glyph_list = self.tablets[tablet][line].split('-')
                for glyph in glyph_list:
                    if glyph != '000':
                        if glyph not in glyph_counts:
                            glyph_counts[glyph] = 0
                        glyph_counts[glyph] += 1
        return glyph_counts

    def get_percentages(self):
        glyph_counts = self.get_counts()
        total = sum(glyph_counts.values())
        for glyph in glyph_counts:
            glyph_counts[glyph] /= total
        glyph_list = [glyph for glyph in glyph_counts]
        glyph_list.sort(key=lambda x: glyph_counts[x], reverse=True)
        percentages = [glyph_counts[glyph] for glyph in glyph_list]
        cumulative_percentages = np.cumsum(percentages)
        df = pd.DataFrame(np.stack((glyph_list, percentages, cumulative_percentages), axis=1))
        df.columns = ['Glyph', 'Percent', 'Cumulative Percent']
        return df
    
    def get_top_n(self, n):
        return self.get_percentages()['Glyph'].values[:n]

    def get_matrix(self, max_glyphs=50):
        transition_matrix = np.zeros((max_glyphs+1, max_glyphs+1))
        glyph_counts = self.get_counts()
        glyphs = list(glyph_counts.keys())
        glyphs.sort(key=lambda x: glyph_counts[x], reverse=True)
        glyphs = glyphs[:max_glyphs]
        for tablet in self.tablets:
            for line in self.tablets[tablet]:
                glyph_list = self.tablets[tablet][line].split('-')
                for i in range(len(glyph_list) - 1):
                    glyph = glyph_list[i]
                    next_glyph = glyph_list[i+1]
                    j = glyphs.index(glyph) if glyph in glyphs else max_glyphs
                    k = glyphs.index(next_glyph) if next_glyph in glyphs else max_glyphs
                    transition_matrix[j, k] += 1
        row_total = transition_matrix.sum(axis=1)
        for i in range(len(transition_matrix)):
            if row_total[i]:
                transition_matrix[i] /= row_total[i]
        return transition_matrix
