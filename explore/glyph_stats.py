import numpy as np
import pandas as pd


class GlyphStats:
    def __init__(self, tablets):
        self.tablets = tablets

    def get_counts(self):
        glyph_counts = {}
        for tablet in self.tablets:
            for line in self.tablets[tablet]:
                glyphs = self.tablets[tablet][line].split('-')
                for glyph in glyphs:
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

    def get_matrix(self):
        pass
