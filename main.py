import pickle

from texts import texts
from train import train_model
from glyphs import decode_glyphs
from parallels import parallels_no_200
from gru.seq2seq import *


encoder, decoder, src_data, tgt_data, config = train_model(texts, num_epochs=40)
df, encoded_glyphs = decode_glyphs(parallels_no_200, config, encoder, decoder)
df.to_csv("results.csv")


final_data = {
    'src_data': src_data,
    'tgt_data': tgt_data,
    'encoded_glyphs': encoded_glyphs
}


with open("data/data", "wb") as f:
    pickle.dump(final_data, f)
