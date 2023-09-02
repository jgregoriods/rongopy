import numpy as np
import tensorflow as tf
import pickle

from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.sequence import pad_sequences

from gru.seq2seq import *
from gru.utils import *


def train_model(texts, num_epochs):
    random.seed(123)
    np.random.seed(123)
    tf.random.set_seed(123)

    RNG = np.random.default_rng(123)

    syllables_no_sp = [split_syllables(sentence) for sentence in texts]
    syllables = [["<s>"] + split_syllables(sentence, keep_spaces=True) + ["</s>"] for sentence in texts]

    src_data = []
    tgt_data = []
    sample_size = len(syllables_no_sp) // 5
    for i in range(len(syllables_no_sp)):
        for j in range(10):
            sample = RNG.choice(syllables_no_sp, sample_size)
            ranks = get_frequencies(sample)
            src_data += encode([syllables_no_sp[i]], ranks)
            tgt_data.append(syllables[i])

    # tokenization
    src_text_tokenized, src_text_tokenizer = tokenize(src_data)
    tgt_text_tokenized, tgt_text_tokenizer = tokenize(tgt_data)

    # vocabulary size
    src_vocab_size = len(src_text_tokenizer.word_index) + 1
    tgt_vocab_size = len(tgt_text_tokenizer.word_index) + 1

    # max sequence lengths
    max_src_len = int(len(max(src_text_tokenized,key=len)))
    max_tgt_len = int(len(max(tgt_text_tokenized,key=len)))

    # padding
    src_pad_sentence = pad_sequences(src_text_tokenized, max_src_len, padding = "post")
    tgt_pad_sentence = pad_sequences(tgt_text_tokenized, max_tgt_len, padding = "post")

    # GRU
    X_train, y_train = shuffle(src_pad_sentence, tgt_pad_sentence, random_state=123)

    BUFFER_SIZE = len(X_train)
    BATCH_SIZE = 32
    N_BATCH = BUFFER_SIZE // BATCH_SIZE
    EPOCHS = num_epochs

    encoder = Encoder(src_vocab_size, 100, 250, BATCH_SIZE)
    decoder = Decoder(tgt_vocab_size, 100, 250, BATCH_SIZE)

    optimizer = tf.optimizers.Adam()

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    config = {
        'batch_size': BATCH_SIZE,
        'src_vocab_size': src_vocab_size,
        'tgt_vocab_size': tgt_vocab_size,
        'max_src_len': max_src_len,
        'max_tgt_len': max_tgt_len,
        'src_text_tokenizer': src_text_tokenizer,
        'tgt_text_tokenizer': tgt_text_tokenizer
    }

    with open("config/config", "wb") as f:
        pickle.dump(config, f)

    train(encoder, decoder, optimizer, dataset, BATCH_SIZE, N_BATCH, EPOCHS)

    return encoder, decoder, src_pad_sentence, tgt_pad_sentence, config
