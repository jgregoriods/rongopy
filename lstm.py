import tensorflow as tf
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, Dense, LSTM
from sklearn.utils import shuffle

from rapanui import real_rapanui, fake_rapanui

real = pd.DataFrame(real_rapanui, columns=['text'])
real['label'] = 0

fake = pd.DataFrame(fake_rapanui, columns=['text'])
fake['label'] = 1

all_texts = pd.concat([real, fake], ignore_index=True)
all_texts = shuffle(all_texts)
all_texts.reset_index(inplace=True)

