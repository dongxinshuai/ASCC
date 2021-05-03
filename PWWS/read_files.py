# -*- coding: utf-8 -*-
import os
import re
import sys
import csv
from tqdm import tqdm
import json
from .config import config
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
#import nltk
#nltk.download('punkt')

try:
    import cPickle as pickle
except ImportError:
    import pickle

def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


def read_snli_files(opt, filetype):
    def label_switch(str):
        if str == "entailment":
            return [1, 0, 0]
        if str == "contradiction":
            return [0, 1, 0]
        if str == "neutral":
            return [0, 0, 1]
        raise NotImplementedError

    split = filetype
    totals = {'train': 550152, 'dev': 10000, 'test': 10000}
    all_prem = []
    all_hypo = []
    all_labels = []

    fn = os.path.join(opt.work_path, 'data_set/snli_1.0/snli_1.0_{}.jsonl'.format(split))
    with open(fn) as f:
        for line in tqdm(f, total=totals[split]):
            example = json.loads(line)
            prem, hypo, gold_label = example['sentence1'], example['sentence2'], example['gold_label']
            try:
                one_hot_label = label_switch(gold_label)

                from nltk import word_tokenize
                prem = ' '.join(word_tokenize(prem))
                hypo = ' '.join(word_tokenize(hypo))

                all_prem.append(prem)
                all_hypo.append(hypo)
                all_labels.append(one_hot_label)

            except:
                continue
    return all_prem, all_hypo, all_labels

def split_snli_files(opt):
    filename = os.path.join(opt.work_path, "temp/split_snli_files")
    if os.path.exists(filename):
        print('Read processed SNLI dataset')
        f=open(filename,'rb')
        saved=pickle.load(f)
        f.close()
        train_perms=saved['train_perms']
        train_hypos=saved['train_hypos']
        train_labels=saved['train_labels']
        test_perms=saved['test_perms']
        test_hypos=saved['test_hypos']
        test_labels=saved['test_labels']
        dev_perms=saved['dev_perms']
        dev_hypos=saved['dev_hypos']
        dev_labels=saved['dev_labels']
    else:
        print('Processing SNLI dataset')
        train_perms, train_hypos, train_labels = read_snli_files(opt, 'train')
        dev_perms, dev_hypos, dev_labels = read_snli_files(opt, 'dev')
        test_perms, test_hypos, test_labels = read_snli_files(opt, 'test')
        f=open(filename,'wb')
        saved={}
        saved['train_perms']=train_perms
        saved['train_hypos']=train_hypos
        saved['train_labels']=train_labels
        saved['test_perms']=test_perms
        saved['test_hypos']=test_hypos
        saved['test_labels']=test_labels
        saved['dev_perms']=dev_perms
        saved['dev_hypos']=dev_hypos
        saved['dev_labels']=dev_labels
        pickle.dump(saved,f)
        f.close()
    return train_perms, train_hypos, train_labels, dev_perms, dev_hypos, dev_labels, test_perms, test_hypos, test_labels


def read_imdb_files(opt, filetype):

    all_labels = []
    for _ in range(12500):
        all_labels.append([0, 1])
    for _ in range(12500):
        all_labels.append([1, 0])

    all_texts = []
    file_list = []
    path = os.path.join(opt.work_path, 'data_set/aclImdb/')
    pos_path = path + filetype + '/pos/'
    for file in os.listdir(pos_path):
        file_list.append(pos_path + file)
    neg_path = path + filetype + '/neg/'
    for file in os.listdir(neg_path):
        file_list.append(neg_path + file)
    for file_name in file_list:
        with open(file_name, 'r', encoding='utf-8') as f:
    
            from nltk import word_tokenize
            x_raw = f.readlines()[0].strip().replace('<br />', ' ')
            x_toks = word_tokenize(x_raw)
            #num_words += len(x_toks)
            all_texts.append(' '.join(x_toks))

            """
            temp = f.readlines()
            temp2 = [re.sub(r'[.]', r'. ', x) for x in temp]
            temp3 = [re.sub(r'[ ][ ]', r' ', x) for x in temp2]
            all_texts.append(rm_tags(" ".join(temp3)))
            """
    return all_texts, all_labels


def split_imdb_files(opt):
    filename = os.path.join(opt.work_path, "temp/split_imdb_files")
    if os.path.exists(filename):
        print('Read processed IMDB dataset')
        f=open(filename,'rb')
        saved=pickle.load(f)
        f.close()
        train_texts=saved['train_texts']
        train_labels=saved['train_labels']
        test_texts=saved['test_texts']
        test_labels=saved['test_labels']
        dev_texts=saved['dev_texts']
        dev_labels=saved['dev_labels']
    else:
        print('Processing IMDB dataset')
        train_texts, train_labels = read_imdb_files(opt, 'train')
        test_texts, test_labels = read_imdb_files(opt, 'test')
        dev_texts = test_texts[12500-500:12500] + test_texts[25000-500:25000]
        dev_labels = test_labels[12500-500:12500] + test_labels[25000-500:25000]
        
        #test_texts = test_texts[:12500-500] + test_texts[12500:25000-500]
        #test_labels = test_labels[:12500-500] + test_labels[12500:25000-500]

        test_texts = test_texts[:12500] + test_texts[12500:25000]
        test_labels = test_labels[:12500] + test_labels[12500:25000]

        f=open(filename,'wb')
        saved={}
        saved['train_texts']=train_texts
        saved['train_labels']=train_labels
        saved['test_texts']=test_texts
        saved['test_labels']=test_labels
        saved['dev_texts']=dev_texts
        saved['dev_labels']=dev_labels
        pickle.dump(saved,f)
        f.close()
    return train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels


def read_yahoo_files():
    text_data_dir = './PWWS/data_set/yahoo_10'

    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(text_data_dir)):
        path = os.path.join(text_data_dir, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding='latin-1')
                    texts.append(f.read())
                    f.close()
                    labels.append(label_id)

    labels = to_categorical(np.asarray(labels))
    return texts, labels, labels_index


def split_yahoo_files():
    print('Processing Yahoo! Answers dataset')
    texts, labels, _ = read_yahoo_files()
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2)
    return train_texts, train_labels, test_texts, test_labels


def read_agnews_files(filetype):
    texts = []
    labels_index = []  # The index of label of all input sentences, which takes the values 1,2,3,4
    doc_count = 0  # number of input sentences
    path = r'./PWWS/data_set/ag_news_csv/{}.csv'.format(filetype)
    csvfile = open(path, 'r')
    for line in csv.reader(csvfile, delimiter=',', quotechar='"'):
        content = line[1] + ". " + line[2]
        texts.append(content)
        labels_index.append(line[0])
        doc_count += 1

    # Start document processing
    labels = []
    for i in range(doc_count):
        label_class = np.zeros(config.num_classes['agnews'], dtype='float32')
        label_class[int(labels_index[i]) - 1] = 1
        labels.append(label_class)

    return texts, labels, labels_index


def split_agnews_files():
    print("Processing AG's News dataset")
    train_texts, train_labels, _ = read_agnews_files('train')  # 120000
    test_texts, test_labels, _ = read_agnews_files('test')  # 7600
    return train_texts, train_labels, test_texts, test_labels


if __name__ == '__main__':
    split_agnews_files()
