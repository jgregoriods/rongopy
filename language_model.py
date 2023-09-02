from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Laplace

from texts import texts


train_data, padded_sents = padded_everygram_pipeline(2, [i.split() for i in texts])

language_model = Laplace(2)
language_model.fit(train_data, padded_sents)
