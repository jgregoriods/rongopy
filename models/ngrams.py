import numpy as np


class GlyphPred:
	def __init__(self, raw_texts):
		self.texts = self.preprocess(raw_texts)
		self.probs = None

	def preprocess(self, raw_texts):
		texts = []
		for text in raw_texts:
			for line in raw_texts[text]:
				texts.append((raw_texts[text][line].split('-')))
		return texts

	def _normalise_rows(self, d):
		for k in d:
			tot = np.sum(list(d[k].values()))
			for v in d[k]:
				d[k][v] /= tot
		return d

	def train(self, ngram_range):
		d = {}
		for line in self.texts:
			for i in range(ngram_range[0], ngram_range[1]+1):
				for j in range(len(line) - i - 1):
					k, v = ' '.join(line[j:j+i]), line[j+i+1]
					if k not in d:
						d[k] = {}
					if v not in d[k]:
						d[k][v] = 0
					d[k][v] += 1
		self.probs = self._normalise_rows(d)

	def predict(self, input_sequence):
		for i in range(len(input_sequence)):
			try:
				glyphs = self.probs[' '.join(input_sequence[i:len(input_sequence)+1])]
				keys = []
				probs = []
				for k, v in glyphs.items():
					keys.append(k)
					probs.append(v)
				return np.random.choice(keys, 1, False, probs)[0]
			except KeyError:
				continue

	def generate_sequence(self, seed, length):
		seq = seed.split(' ')
		for i in range(length):
			seq.append(self.predict(seq))
		return '-'.join(seq)
