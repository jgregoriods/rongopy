import re

from nltk.lm.preprocessing import padded_everygram_pipeline
from tensorflow.keras.preprocessing.text import Tokenizer


def split_syllables(sentence, keep_spaces=False):
    res = []
    if keep_spaces:
        sentence = sentence.replace(" ", " _ ")
    words = sentence.split()
    for i in range(len(words)):
        word = words[i]

        x = re.sub(r"([aeiouāēīōū])", r"\1|", word)
        x = re.sub(r"n(?![aeiouāēīōū])", r"n|", x)
        x = x.split("|")

        syllables = [j for j in x if j]

        res += syllables
    return res


def get_frequencies(texts, max_rank=30):
    toker = Tokenizer()
    toker.fit_on_texts(texts)
    char_frequencies = sorted(list(toker.word_counts), key=lambda x: toker.word_counts[x], reverse=True)
    res = {}
    for char in char_frequencies[:max_rank]:
        res[char] = str(char_frequencies.index(char)+1)
    for char in char_frequencies[max_rank:]:
        res[char] = "0"
    #return {char: str(min(char_frequencies.index(char)+1, max_rank)) for char in char_frequencies}
    return res


def encode(texts, ranks):
    #max_rank = max([str(i) for i in list(ranks.values())])
    res = []
    for text in texts:
        encoded = []
        for syl in text:
            if syl in ranks:
                encoded.append(ranks[syl])
            else:
                encoded.append("0")
        res.append(encoded)
    return res


def tokenize(texts):
    toker = Tokenizer()
    toker.fit_on_texts(texts)
    return toker.texts_to_sequences(texts), toker


def perplexity(sentences, model):
    res = []
    test_data, _ = padded_everygram_pipeline(model.order, [i.split() for i in sentences])
    for i,test in enumerate(test_data):
        res.append(model.perplexity(test))
    return res


def clean_char(x):
    syms = "bcdeghijkostxy!*?()"
    for s in syms:
        x = x.replace(s, "")
    x = x.replace(".", " ").split()
    for i in range(len(x)):
        if ":" in x[i]:
            j = x[i].index(":")
            x[i] = x[i][j+1:] + " " + x[i][:j]
    x = ' '.join([i.lstrip("0") for i in ' '.join(x).split()]).lower()
    if x != "0":
        return x
    else:
        return ""
