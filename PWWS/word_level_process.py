# -*- coding: utf-8 -*-
import spacy
import os
import re
from keras.preprocessing import sequence
#from keras.preprocessing.text import Tokenizer
from .tokenizer_for_spacy import Tokenizer
import numpy as np
from .read_files import split_imdb_files, split_yahoo_files, split_agnews_files, split_snli_files
from .read_files import read_imdb_files, read_yahoo_files, read_agnews_files, read_snli_files
from .config import config

try:
    import cPickle as pickle
except ImportError:
    import pickle

nlp = spacy.load('en_core_web_sm')

global imdb_tokenizer, yahoo_tokenizer, agnews_tokenizer, snli_tokenizer
imdb_tokenizer=yahoo_tokenizer=agnews_tokenizer=snli_tokenizer=None


def update_tokenizer(dataset, tokenizer):

    if dataset == 'imdb':
        global imdb_tokenizer
        imdb_tokenizer = tokenizer
    elif dataset == 'snli':
        global snli_tokenizer
        snli_tokenizer = tokenizer
    elif dataset == 'yahoo':
        global yahoo_tokenizer
        yahoo_tokenizer = tokenizer
    elif dataset == 'agnews':
        global agnews_tokenizer
        agnews_tokenizer = tokenizer

def get_tokenizer(opt):
    dataset = opt.dataset
    texts = None
    if dataset == 'imdb':
        global imdb_tokenizer
        if imdb_tokenizer is not None:
            return imdb_tokenizer

        imdb_tokenizer_file = os.path.join(opt.work_path,"temp/imdb_tokenizer.pickle")
        if os.path.exists(imdb_tokenizer_file):
            f=open(imdb_tokenizer_file,'rb')
            imdb_tokenizer=pickle.load(f)
            f.close()
        else:
            train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = split_imdb_files(opt)

            imdb_tokenizer = Tokenizer(num_words=config.num_words[dataset], use_spacy=False)
            imdb_tokenizer.fit_on_texts(train_texts)
            f=open(imdb_tokenizer_file,'wb')
            pickle.dump(imdb_tokenizer, f)
            f.close()
        return imdb_tokenizer
    elif dataset == 'snli':
        global snli_tokenizer
        if snli_tokenizer is not None:
            return snli_tokenizer

        snli_tokenizer_file = os.path.join(opt.work_path,"temp/snli_tokenizer.pickle")
        if os.path.exists(snli_tokenizer_file):
            f=open(snli_tokenizer_file,'rb')
            snli_tokenizer=pickle.load(f)
            f.close()
        else:
            train_perms, train_hypos, train_labels, dev_perms, dev_hypos, dev_labels, test_perms, test_hypos, test_labels = split_snli_files(opt)
            snli_tokenizer = Tokenizer(num_words=config.num_words[dataset], use_spacy=False)
            snli_tokenizer.fit_on_texts( train_perms+train_hypos )
            f=open(snli_tokenizer_file,'wb')
            pickle.dump(snli_tokenizer, f)
            f.close()
        return snli_tokenizer


def text_process_for_single(tokenizer, texts, dataset):
    maxlen = config.word_max_len[dataset]
    seq = tokenizer.texts_to_sequences(texts)
    seq = sequence.pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')
    return seq

def text_process_for_single_bert(tokenizer,texts, dataset):
    maxlen = config.word_max_len[dataset]

    res=[]

    for text in texts:
        encoded = tokenizer.encode_plus(
                text, 
                None,
                add_special_tokens = True,
                max_length = maxlen,
                pad_to_max_length = True,
            )
        res.append(encoded)

    return res

def label_process_for_single(tokenizer, labels, dataset):
    maxlen = config.word_max_len[dataset]

    out = np.array(labels)
    return out

def word_process(train_texts, train_labels, test_texts, test_labels, dataset):
    maxlen = config.word_max_len[dataset]
    tokenizer = get_tokenizer(opt)

    x_train_seq = tokenizer.texts_to_sequences(train_texts)
    x_test_seq = tokenizer.texts_to_sequences(test_texts)
    x_train = sequence.pad_sequences(x_train_seq, maxlen=maxlen, padding='post', truncating='post')
    x_test = sequence.pad_sequences(x_test_seq, maxlen=maxlen, padding='post', truncating='post')
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    return x_train, y_train, x_test, y_test


def text_to_vector(text, tokenizer, dataset):
    maxlen = config.word_max_len[dataset]
    vector = tokenizer.texts_to_sequences([text])
    vector = sequence.pad_sequences(vector, maxlen=maxlen, padding='post', truncating='post')
    return vector


def text_to_vector_for_all(text_list, tokenizer, dataset):
    maxlen = config.word_max_len[dataset]
    vector = tokenizer.texts_to_sequences(text_list)
    vector = sequence.pad_sequences(vector, maxlen=maxlen, padding='post', truncating='post')
    return vector

