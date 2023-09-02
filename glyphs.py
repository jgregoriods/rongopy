import pandas as pd

from Levenshtein import distance as lev

from gru.seq2seq import *
from gru.utils import *
from texts import texts
from language_model import language_model


def decode_glyphs(sequences, config, encoder=None, decoder=None):
    glyph_freqs = get_frequencies(sequences)
    encoded_glyphs = encode(sequences, glyph_freqs)

    batch_size = config['batch_size']
    src_vocab_size = config['src_vocab_size']
    tgt_vocab_size = config['tgt_vocab_size']
    max_src_len = config['max_src_len']
    max_tgt_len = config['max_tgt_len']
    src_text_tokenizer = config['src_text_tokenizer']
    tgt_text_tokenizer = config['tgt_text_tokenizer']

    if encoder is None:
        encoder = Encoder(src_vocab_size, 100, 250, batch_size)
        encoder.load_weights("models/encoder")
    
    if decoder is None:
        decoder = Decoder(tgt_vocab_size, 100, 250, batch_size)
        decoder.load_weights("models/decoder")

    preds = []
    for i in range(len(encoded_glyphs)):
        pred, _, att = evaluate(encoded_glyphs[i], encoder, decoder, max_src_len, max_tgt_len, src_text_tokenizer, tgt_text_tokenizer)
        preds.append(pred.replace(" ", "").replace("</s>", "").replace("_", " "))
    
    sents = []
    for i in range(len(preds)):
        sents.append(texts[np.argmin([lev(preds[i], texts[j]) for j in range(len(texts))])])

    scores = perplexity(preds, language_model)
    df = pd.DataFrame({'Glyphs': sequences, 'Encoded': encoded_glyphs, 'Prediction': preds, 'Perplexity': scores, 'BestMatch': sents})

    return df, encoded_glyphs

