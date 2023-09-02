import pickle

from gru.seq2seq import *
from parallels import parallels_no_200


with open('config/config', 'rb') as f:
    config = pickle.load(f)

with open('data/data', 'rb') as f:
    final_data = pickle.load(f)


batch_size = config['batch_size']
src_vocab_size = config['src_vocab_size']
tgt_vocab_size = config['tgt_vocab_size']
max_src_len = config['max_src_len']
max_tgt_len = config['max_tgt_len']
src_text_tokenizer = config['src_text_tokenizer']
tgt_text_tokenizer = config['tgt_text_tokenizer']


src_data = final_data['src_data']
tgt_data = final_data['tgt_data']
encoded_glyphs = final_data['encoded_glyphs']


encoder = Encoder(src_vocab_size, 100, 250, batch_size)
decoder = Decoder(tgt_vocab_size, 100, 250, batch_size)

encoder.load_weights('models/encoder')
decoder.load_weights('models/decoder')

idx = 10
pred, _, att = evaluate(encoded_glyphs[idx], encoder, decoder, max_src_len, max_tgt_len, src_text_tokenizer, tgt_text_tokenizer)
plot_attention(att, parallels_no_200[idx], pred.split())
