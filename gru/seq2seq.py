# Copyright 2018 The TensorFlow Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
# https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/eager/python/examples/nmt_with_attention/nmt_with_attention.ipynb


import matplotlib.pyplot as plt
import time
import numpy as np
import tensorflow as tf
import gc
import random

from tensorflow.keras.preprocessing.sequence import pad_sequences


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True, 
                                       return_state=True, 
                                       recurrent_activation='sigmoid')
        
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)        
        return output, state
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True, 
                                       return_state=True, 
                                       recurrent_activation='sigmoid')
        self.fc = tf.keras.layers.Dense(vocab_size)
        
        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))
        
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        
        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)
        
        return x, state, attention_weights
        
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))


def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)


def train(encoder, decoder, optimizer, dataset, batch_size, n_batch, epochs, patience=10):
    min_loss = None
    no_improv_steps = 0
    for epoch in range(epochs):
        start = time.time()
        
        hidden = encoder.initialize_hidden_state()
        total_loss = 0
        
        for (batch, (inp, targ)) in enumerate(dataset):
            loss = 0
            
            with tf.GradientTape() as tape:
                enc_output, enc_hidden = encoder(inp, hidden)
                
                dec_hidden = enc_hidden
                
                dec_input = tf.expand_dims([0] * batch_size, 1)       
                
                for t in range(1, targ.shape[1]):
                    predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                    
                    loss += loss_function(targ[:, t], predictions)

                    dec_input = tf.expand_dims(targ[:, t], 1)
            
            batch_loss = (loss / int(targ.shape[1]))
            
            total_loss += batch_loss
            
            variables = encoder.variables + decoder.variables
            
            gradients = tape.gradient(loss, variables)
            
            optimizer.apply_gradients(zip(gradients, variables))
      
        print('Epoch {}/{} Loss {:.4f}'.format(epoch + 1, epochs,
                                            total_loss / n_batch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

        cur_loss = total_loss / n_batch

        if epoch == 0:
            min_loss = cur_loss
            encoder.save_weights("models/encoder")
            decoder.save_weights("models/decoder")
        else:
            if cur_loss > min_loss:
                no_improv_steps += 1
            else:
                encoder.save_weights("models/encoder")
                decoder.save_weights("models/decoder")
                no_improv_steps = 0
                min_loss = cur_loss

    print("Restoring best weights.")
    encoder.load_weights("models/encoder")
    decoder.load_weights("models/decoder")


def get_token(i, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == i:
            return word
    return None


def evaluate(sentence, encoder, decoder, max_length_inp, max_length_targ, src_text_tokenizer, tgt_text_tokenizer):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    toked = src_text_tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(toked, max_length_inp, padding = "post")
    inputs = tf.convert_to_tensor(padded)
    
    result = ''

    hidden = [tf.zeros((1, 250))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([0], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += ' ' + get_token(predicted_id, tgt_text_tokenizer)

        if get_token(predicted_id, tgt_text_tokenizer) == '</s>':
            attention_plot = attention_plot[:len(result.split()), :len(sentence)]
            return result.strip(), inputs, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result.split()), :len(sentence)]
    return result.strip(), inputs, attention_plot


def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    
    fontdict = {'fontsize': 14}
    
    ax.set_xticks(list(range(len(sentence))))
    ax.set_yticks(list(range(len(predicted_sentence))))

    ax.set_xticklabels(sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels(predicted_sentence, fontdict=fontdict)

    plt.show()
