#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 09:33:46 2019

@author: chayan
"""
#import tensorflow as tf
#from tensorflow.contrib import keras
#L = keras.layers
#import vocab_utils as vutil
#
#vocab = vutil.get_vocab()
#
#IMG_SIZE = 299
#IMG_EMBED_SIZE = 2048
#IMG_EMBED_BOTTLENECK = 120
#WORD_EMBED_SIZE = 100
#LSTM_UNITS = 300
#LOGIT_BOTTLENECK = 120
#pad_idx = vocab[vutil.PAD]

#class decoder:
#    # [batch_size, IMG_EMBED_SIZE] of CNN image features
#    img_embeds = tf.placeholder('float32', [None, IMG_EMBED_SIZE])
#    
#    # [batch_size, time steps] of word ids
#    sentences = tf.placeholder('int32', [None, None])
#    
#    # we use bottleneck here to reduce the number of parameters
#    # image embedding -> bottleneck
#    img_embed_to_bottleneck = L.Dense(IMG_EMBED_BOTTLENECK, 
#                                      input_shape=(None, IMG_EMBED_SIZE), 
#                                      activation='elu')
#    
#    # image embedding bottleneck -> lstm initial state
#    img_embed_bottleneck_to_h0 = L.Dense(LSTM_UNITS,
#                                         input_shape=(None, IMG_EMBED_BOTTLENECK),
#                                         activation='elu')
#    
#    # word -> embedding
#    word_embed = L.Embedding(len(vocab), WORD_EMBED_SIZE)
#    
#    # lstm cell (from tensorflow)
#    lstm = tf.nn.rnn_cell.LSTMCell(LSTM_UNITS)
#    
#    # we use bottleneck here to reduce model complexity
#    # lstm output -> logits bottleneck
#    token_logits_bottleneck = L.Dense(LOGIT_BOTTLENECK, 
#                                      input_shape=(None, LSTM_UNITS),
#                                      activation="elu")
#    
#    # logits bottleneck -> logits for next token prediction
#    token_logits = L.Dense(len(vocab),
#                           input_shape=(None, LOGIT_BOTTLENECK))
#    
#    # initial lstm cell state of shape (None, LSTM_UNITS),
#    # we need to condition it on `img_embeds` placeholder.
#    c0 = h0 = img_embed_bottleneck_to_h0(img_embed_to_bottleneck(img_embeds))
#
#    # embed all tokens but the last for lstm input,
#    # remember that L.Embedding is callable,
#    # use `sentences` placeholder as input.
#    word_embeds = word_embed(sentences[:, :-1])
#    
#    # during training we use ground truth tokens `word_embeds` as context for next token prediction.
#    # that means that we know all the inputs for our lstm and can get 
#    # all the hidden states with one tensorflow operation (tf.nn.dynamic_rnn).
#    # `hidden_states` has a shape of [batch_size, time steps, LSTM_UNITS].
#    hidden_states, _ = tf.nn.dynamic_rnn(lstm, word_embeds,
#                                         initial_state=tf.nn.rnn_cell.LSTMStateTuple(c0, h0))
#
#    # now we need to calculate token logits for all the hidden states
#    
#    # first, we reshape `hidden_states` to [-1, LSTM_UNITS]
#    flat_hidden_states = tf.reshape(hidden_states, [-1, LSTM_UNITS]) ### YOUR CODE HERE ###
#
#    # then, we calculate logits for next tokens using `token_logits_bottleneck` and `token_logits` layers
#    ### YOUR CODE HERE ###
#    flat_token_logits = token_logits(token_logits_bottleneck(flat_hidden_states))
#    
#    # then, we flatten the ground truth token ids.
#    # remember, that we predict next tokens for each time step,
#    # use `sentences` placeholder.
#    flat_ground_truth = tf.reshape(sentences[:, 1:], [-1]) ### YOUR CODE HERE ###
#
#    # we need to know where we have real tokens (not padding) in `flat_ground_truth`,
#    # we don't want to propagate the loss for padded output tokens,
#    # fill `flat_loss_mask` with 1.0 for real tokens (not pad_idx) and 0.0 otherwise.
#    
#    flat_loss_mask = tf.map_fn(lambda idx: tf.cond(tf.equal(idx, pad_idx), lambda: 0.0, lambda: 1.0), 
#                               flat_ground_truth, dtype='float')
#
#    # compute cross-entropy between `flat_ground_truth` and `flat_token_logits` predicted by lstm
#    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
#        labels=flat_ground_truth, 
#        logits=flat_token_logits
#    )
#
#    # compute average `xent` over tokens with nonzero `flat_loss_mask`.
#    # we don't want to account misclassification of PAD tokens, because that doesn't make sense,
#    # we have PAD tokens for batching purposes only!
#    masked_xent = tf.multiply(xent, flat_loss_mask)
#    loss_sum = tf.reduce_sum(masked_xent)
#    non_zero_count = tf.cast(tf.math.count_nonzero(masked_xent), tf.float32)
#    loss = tf.divide(loss_sum, non_zero_count)
