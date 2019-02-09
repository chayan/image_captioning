#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 13:31:44 2019

@author: chayan
"""
import itertools
import re
import utils
import cv2
import zipfile
import json
from collections import defaultdict

train_img_fns = utils.read_pickle("train_img_fns.pickle")
val_img_fns = utils.read_pickle("val_img_fns.pickle")

def get_captions_for_fns(fns, zip_fn, zip_json_path):
    zf = zipfile.ZipFile(zip_fn)
    j = json.loads(zf.read(zip_json_path).decode("utf8"))
    id_to_fn = {img["id"]: img["file_name"] for img in j["images"]}
    fn_to_caps = defaultdict(list)
    for cap in j['annotations']:
        fn_to_caps[id_to_fn[cap['image_id']]].append(cap['caption'])
    fn_to_caps = dict(fn_to_caps)
    return list(map(lambda x: fn_to_caps[x], fns))
    
train_captions = get_captions_for_fns(train_img_fns, "captions_train-val2014.zip", 
                                      "annotations/captions_train2014.json")

val_captions = get_captions_for_fns(val_img_fns, "captions_train-val2014.zip", 
                                      "annotations/captions_val2014.json")

# special tokens
PAD = "#PAD#"
UNK = "#UNK#"
START = "#START#"
END = "#END#"

# split sentence into tokens (split into lowercased words)
def split_sentence(sentence):
    return list(filter(lambda x: len(x) > 0, re.split('\W+', sentence.lower())))

def get_vocab():
    """
    Return {token: index} for all train tokens (words) that occur 5 times or more, 
        `index` should be from 0 to N, where N is a number of unique tokens in the resulting dictionary.
    Use `split_sentence` function to split sentence into tokens.
    Also, add PAD (for batch padding), UNK (unknown, out of vocabulary), 
        START (start of sentence) and END (end of sentence) tokens into the vocabulary.
    """
    
    ### YOUR CODE HERE ###
    all_tokens = itertools.chain.from_iterable([split_sentence(caption) for captions in train_captions for caption in captions])
    tokens_dict = defaultdict(int)
    for token in all_tokens:
      tokens_dict[token] += 1
    vocab = list(filter(lambda token: tokens_dict[token] >= 5, tokens_dict.keys())) + [PAD, UNK, START, END]
    return {token: index for index, token in enumerate(sorted(vocab))}

def index_sentence(sentence, vocab):
    tokens = split_sentence(sentence)
    tokens = [START] + tokens + [END]
    return [vocab.get(token, vocab[UNK]) for token in tokens]

def caption_tokens_to_indices(captions, vocab):
    """
    `captions` argument is an array of arrays:
    [
        [
            "image1 caption1",
            "image1 caption2",
            ...
        ],
        [
            "image2 caption1",
            "image2 caption2",
            ...
        ],
        ...
    ]
    Use `split_sentence` function to split sentence into tokens.
    Replace all tokens with vocabulary indices, use UNK for unknown words (out of vocabulary).
    Add START and END tokens to start and end of each sentence respectively.
    For the example above you should produce the following:
    [
        [
            [vocab[START], vocab["image1"], vocab["caption1"], vocab[END]],
            [vocab[START], vocab["image1"], vocab["caption2"], vocab[END]],
            ...
        ],
        ...
    ]
    """

    return [[index_sentence(sentence, vocab) for sentence in sentences] for sentences in captions]


    