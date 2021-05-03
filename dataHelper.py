# -*- coding: utf-8 -*-

import os
import numpy as np
import string
from collections import Counter
import pandas as pd
from tqdm import tqdm
import random
import time

import torch
from torch.autograd import Variable

from PWWS.read_files import split_imdb_files, split_yahoo_files, split_agnews_files, split_snli_files
from PWWS.word_level_process import word_process, get_tokenizer, update_tokenizer, text_process_for_single, label_process_for_single, text_process_for_single_bert
from PWWS.neural_networks import get_embedding_index, get_embedding_matrix

from PWWS.paraphrase import generate_synonym_list_from_word, generate_synonym_list_by_dict, get_syn_dict
from PWWS.config import config

from torchvision.datasets.vision import VisionDataset

import torch.utils.data

from codecs import open
try:
    import cPickle as pickle
except ImportError:
    import pickle
class Alphabet(dict):
    def __init__(self, start_feature_id = 1, alphabet_type="text"):
        self.fid = start_feature_id
        if alphabet_type=="text":
            self.add('[PADDING]')
            self.add('[UNK]')
            self.add('[END]')
            self.unknow_token = self.get('[UNK]')
            self.end_token = self.get('[END]')
            self.padding_token = self.get('[PADDING]')

    def add(self, item):
        idx = self.get(item, None)
        if idx is None:
            idx = self.fid
            self[item] = idx
      # self[idx] = item
            self.fid += 1
        return idx
    
    def addAll(self,words):
        for word in words:
            self.add(word)
            
    def dump(self, fname,path="temp"):
        if not os.path.exists(path):
            os.mkdir(path)
        with open(os.path.join(path,fname), "w",encoding="utf-8") as out:
            for k in sorted(self.keys()):
                out.write("{}\t{}\n".format(k, self[k]))

class DottableDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self
        self.allowDotting()
    def allowDotting(self, state=True):
        if state:
            self.__dict__ = self
        else:
            self.__dict__ = dict()
            
class BucketIterator_PWWS(object):
    def __init__(self,x,y,z,opt=None,batch_size=2,shuffle=True,test=False,position=False):
        self.shuffle=shuffle
        self.x = x
        self.y = np.argmax(y, axis=-1)
        self.z = z
        self.batch_size=batch_size
        self.test=test        
        if opt is not None:
            self.setup(opt)
    def setup(self,opt):
        
        self.batch_size=opt.batch_size
        self.shuffle=opt.__dict__.get("shuffle",self.shuffle)
        self.position=opt.__dict__.get("position",False)
        if self.position:
            self.padding_token =  opt.alphabet.padding_token
    
    def transform(self,batch_x,batch_y,batch_z):
        if torch.cuda.is_available():
            #data=data.reset_index()
            text= Variable(torch.LongTensor(batch_x).cuda())
            label= Variable(torch.LongTensor(batch_y).cuda())
            ori_text = batch_z
        else:
            #data=data.reset_index()
            text= Variable(torch.LongTensor(batch_x))
            label= Variable(torch.LongTensor(batch_y))
            ori_text = batch_z
        if self.position:
            position_tensor = self.get_position(data.text)
            return DottableDict({"text":(text,position_tensor),"label":label,"ori_text":ori_text})
        return DottableDict({"text":text,"label":label, "ori_text":ori_text})
    
    def get_position(self,inst_data):
        inst_position = np.array([[pos_i+1 if w_i != self.padding_token else 0 for pos_i, w_i in enumerate(inst)] for inst in inst_data])
        inst_position_tensor = Variable( torch.LongTensor(inst_position), volatile=self.test) 
        if torch.cuda.is_available():
            inst_position_tensor=inst_position_tensor.cuda()
        return inst_position_tensor

    def __iter__(self):
        if self.shuffle:
            import random 
            seed = random.randint(0,100)

            random.seed(seed)
            self.x=list(self.x)
            random.shuffle(self.x)
            self.x=np.array(self.x)

            random.seed(seed)
            self.y=list(self.y)
            random.shuffle(self.y)
            self.y=np.array(self.y)

            random.seed(seed)
            random.shuffle(self.z)

        batch_nums = int(len(self.x)/self.batch_size)
        for  i in range(batch_nums):
            start=i*self.batch_size
            end=(i+1)*self.batch_size
            yield self.transform(self.x[start:end],self.y[start:end],self.z[start:end])
        yield self.transform(self.x[-1*self.batch_size:],self.y[-1*self.batch_size:],self.z[-1*self.batch_size:])
    


class SynthesizedData(torch.utils.data.Dataset):
    
    def __init__(self, opt, x, y, syn_data):
        super(SynthesizedData, self).__init__()
        self.x = x.copy()
        self.y = y.copy()
        self.syn_data = syn_data.copy()

        for x in range(len(self.syn_data)):
            self.syn_data[x] = [syn_word for syn_word in self.syn_data[x] if syn_word!=x]

        self.len_voc = len(self.syn_data)+1

    def transform(self, sent, label, anch, pos, neg, anch_valid):
       
        return torch.tensor(sent,dtype=torch.long), torch.tensor(label,dtype=torch.long),torch.tensor(anch,dtype=torch.long),torch.tensor(pos,dtype=torch.long),torch.tensor(neg,dtype=torch.long),torch.tensor(anch_valid,dtype=torch.float)

    def __getitem__(self, index, max_num_anch_per_sent=100, num_pos_per_anch=20, num_neg_per_anch=100):
        sent = self.x[index]
        label = self.y[index].argmax()

        #for x in sent:
        #    self.syn_data[x] = [syn_word for syn_word in self.syn_data[x] if syn_word!=x]
        #try:
        sent_for_anch = [x for x in sent if x>0 and x<len(self.syn_data) and len(self.syn_data[x]) != 0]
        #except:
        #    print(index)
        #while(len(sent_for_anch) < max_num_anch_per_sent):
        #    sent_for_anch.extend(sent_for_anch)
        
        if len(sent_for_anch) > max_num_anch_per_sent:
            anch = random.sample(sent_for_anch, max_num_anch_per_sent)
        else:
            anch = sent_for_anch

        anch_valid = [1 for x in anch]

        pos = []
        neg = []
        for word in anch:
            syn_set = set(self.syn_data[word])
            if len(self.syn_data[word]) == 0:
                pos.append([word for i in range(num_pos_per_anch)])
            elif len(self.syn_data[word]) < num_pos_per_anch:
                    temp = []
                    for i in range( int(num_pos_per_anch/len(self.syn_data[word])) + 1):
                        temp.extend(self.syn_data[word])
                    #pos.append(temp[:num_pos_per_anch])
                    pos.append(random.sample(temp, num_pos_per_anch))
            elif len(self.syn_data[word]) >= num_pos_per_anch:
                pos.append(random.sample(self.syn_data[word], num_pos_per_anch))

            count=0
            temp = []
            while (count<num_neg_per_anch):
                while (1):
                    neg_word = random.randint(0, self.len_voc)
                    if neg_word not in syn_set:
                        break
                temp.append(neg_word)
                count+=1
            neg.append(temp)

        while(len(anch)<max_num_anch_per_sent):
            anch.append(0)
            anch_valid.append(0)
            pos.append([0 for i in range(num_pos_per_anch)])
            neg.append([0 for i in range(num_neg_per_anch)])

        return self.transform(sent, label, anch, pos, neg, anch_valid)

    def __len__(self):
        return len(self.y)


class SynthesizedData_TextLikeSyn(SynthesizedData):
    
    def __init__(self, opt, x, y, syn_data):
        super(SynthesizedData_TextLikeSyn, self).__init__(opt, x, y, syn_data)

    def transform(self, sent, label, anch, pos, neg, anch_valid, text_like_syn, text_like_syn_valid):
       
        return torch.tensor(sent,dtype=torch.long), torch.tensor(label,dtype=torch.long), torch.tensor(anch,dtype=torch.long),torch.tensor(pos,dtype=torch.long),torch.tensor(neg,dtype=torch.long), torch.tensor(anch_valid,dtype=torch.float), torch.tensor(text_like_syn,dtype=torch.long), torch.tensor(text_like_syn_valid,dtype=torch.float),

    def __getitem__(self, index, max_num_anch_per_sent=100, num_pos_per_anch=20, num_neg_per_anch=100, num_text_like_syn=20):
        sent = self.x[index]
        label = self.y[index].argmax()

        text_like_syn=[]
        text_like_syn_valid=[]
        for x in sent:
            text_like_syn_valid.append([])
            if x<len(self.syn_data):
                text_like_syn.append( self.syn_data[x].copy() )
            else:
                text_like_syn.append([])

        for i, x in enumerate(sent):
            temp = text_like_syn[i]
            len_temp=len(temp)
            if len_temp==0:
                text_like_syn_valid[i] = [1]
                text_like_syn_valid[i].extend([0 for times in range(num_text_like_syn-1)])
                text_like_syn[i] = [x]
                text_like_syn[i].extend([0 for times in range(num_text_like_syn-1)])
            elif len_temp>=num_text_like_syn-1:
                temp = random.sample(temp, num_text_like_syn-1)
                temp.append(x)
                text_like_syn[i]=temp
                text_like_syn_valid[i] = [1 for times in range(num_text_like_syn)]
                assert(len(text_like_syn[i])==num_text_like_syn)
            else:
                temp.append(x)
                text_like_syn_valid[i] = [1 for times in range(len(temp))]
                while(len(temp)<num_text_like_syn):
                    temp.append(0)
                    text_like_syn_valid[i].append(0)
                text_like_syn[i]=temp
                assert(len(text_like_syn[i])==num_text_like_syn)

        #for x in sent:
        #    self.syn_data[x] = [syn_word for syn_word in self.syn_data[x] if syn_word!=x]
        #try:
        sent_for_anch = [x for x in sent if x>0 and x<len(self.syn_data) and len(self.syn_data[x]) != 0]
        #except:
        #    print(index)
        #while(len(sent_for_anch) < max_num_anch_per_sent):
        #    sent_for_anch.extend(sent_for_anch)
        
        if len(sent_for_anch) > max_num_anch_per_sent:
            anch = random.sample(sent_for_anch, max_num_anch_per_sent)
        else:
            anch = sent_for_anch

        anch_valid = [1 for x in anch]

        pos = []
        neg = []
        for word in anch:
            syn_set = set(self.syn_data[word].copy())
            if len(self.syn_data[word]) == 0:
                pos.append([word for i in range(num_pos_per_anch)])
            elif len(self.syn_data[word]) < num_pos_per_anch:
                    temp = []
                    for i in range( int(num_pos_per_anch/len(self.syn_data[word])) + 1):
                        temp.extend(self.syn_data[word].copy())
                    #pos.append(temp[:num_pos_per_anch])
                    pos.append(random.sample(temp, num_pos_per_anch))
            elif len(self.syn_data[word]) >= num_pos_per_anch:
                pos.append(random.sample(self.syn_data[word].copy(), num_pos_per_anch))

            count=0
            temp = []
            while (count<num_neg_per_anch):
                while (1):
                    neg_word = random.randint(0, self.len_voc)
                    if neg_word not in syn_set:
                        break
                temp.append(neg_word)
                count+=1
            neg.append(temp)

        while(len(anch)<max_num_anch_per_sent):
            anch.append(0)
            anch_valid.append(0)
            pos.append([0 for i in range(num_pos_per_anch)])
            neg.append([0 for i in range(num_neg_per_anch)])

        return self.transform(sent, label, anch, pos, neg, anch_valid, text_like_syn, text_like_syn_valid)


class SynthesizedData_TextLikeSyn_Bert(SynthesizedData):
    
    def __init__(self, opt, x, y, syn_data, seq_max_len, tokenizer, attack_surface=None, given_y=None):
        self.x = x.copy()
        self.y = y.copy() #y is onehot
        self.syn_data = syn_data.copy()
        self.seq_max_len = seq_max_len
        self.tokenizer = tokenizer

        self.dataset_ids = []
        for i in range(len(x)):
            self.dataset_ids.append(i)

        label = [y.argmax() for y in self.y]

        # for sample idx
        self.cls_positive = [[] for i in range(opt.label_size)]
        for i in range(len(x)):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(opt.label_size)]
        for i in range(opt.label_size):
            for j in range(opt.label_size):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(opt.label_size)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(opt.label_size)]
        #

        if attack_surface is not None:
            xx=[]
            yy=[]
            did=[]
            for i, text in enumerate(self.x):
                if attack_surface.check_in(text.split(' ')):
                    xx.append(text)
                    yy.append(self.y[i])
                    did.append(self.dataset_ids[i])
            self.x = xx
            self.y = yy
            self.dataset_ids = did

        if given_y is not None:
            xx=[]
            yy=[]
            did=[]
            for i, label in enumerate(self.y):
                if self.y[i].argmax()==given_y:
                    xx.append(self.x[i])
                    yy.append(self.y[i])
                    did.append(self.dataset_ids[i])
            self.x = xx
            self.y = yy
            self.dataset_ids = did

        #for x in self.syn_data:
        #    self.syn_data[x] = [syn_word for syn_word in self.syn_data[x] if syn_word!=x]

        #self.len_voc = len(self.syn_data)+1
        #super(SynthesizedData_TextLikeSyn_Bert, self).__init__(opt, x, y, syn_data)

    def transform(self, sent, label, text_like_syn, text_like_syn_valid, mask, token_type_ids, dataset_id, sample_idx):
       
        return torch.tensor(sent,dtype=torch.long), torch.tensor(label,dtype=torch.long), torch.tensor(text_like_syn,dtype=torch.long), torch.tensor(text_like_syn_valid,dtype=torch.float), torch.tensor(mask, dtype = torch.long), torch.tensor(token_type_ids, dtype = torch.long), torch.tensor(dataset_id, dtype = torch.long), torch.tensor(sample_idx, dtype = torch.long)

    def __getitem__(self, index, num_text_like_syn=10, K=1000):

        encoded = self.tokenizer.encode_plus(self.x[index], None, add_special_tokens = True, max_length = self.seq_max_len, pad_to_max_length = True)

        sent = encoded["input_ids"]
        mask = encoded["attention_mask"]
        token_type_ids = encoded["token_type_ids"]

        text_like_syn=[]
        text_like_syn_valid=[]
        for token in sent:
            text_like_syn_valid.append([])
            text_like_syn.append([token])

        splited_words = self.x[index].split()

        for i in range(min(self.seq_max_len-2, len(splited_words))):
            word = splited_words[i]
            if word in self.syn_data:
                text_like_syn[i+1].extend(self.syn_data[word])

        label = self.y[index].argmax()
        dataset_id = self.dataset_ids[index]

        for i, x in enumerate(sent):
            text_like_syn_valid[i] = [1 for times in range(len(text_like_syn[i]))]
            
            while(len(text_like_syn[i])<num_text_like_syn):
                text_like_syn[i].append(0)
                text_like_syn_valid[i].append(0)

            assert(len(text_like_syn[i])==num_text_like_syn)
            assert(len(text_like_syn_valid[i])==num_text_like_syn)


        replace = True if K > len(self.cls_negative[label]) else False
        neg_idx = np.random.choice(self.cls_negative[label], K, replace=replace)
        sample_idx = np.hstack((np.asarray([dataset_id]), neg_idx))

        return self.transform(sent, label, text_like_syn, text_like_syn_valid, mask, token_type_ids, dataset_id, sample_idx)

class SNLI_SynthesizedData_TextLikeSyn_Bert(SynthesizedData):
    
    def __init__(self, opt, perm, hypo, y, syn_data, seq_max_len, tokenizer, attack_surface=None, given_y=None):
        self.perm = perm.copy()
        self.hypo = hypo.copy()
        self.y = y.copy()
        self.syn_data = syn_data.copy()
        self.seq_max_len = seq_max_len
        self.tokenizer = tokenizer

        """
        if attack_surface is not None:
            xx=[]
            yy=[]
            for i, text in enumerate(self.x):
                if attack_surface.check_in(text.split(' ')):
                    xx.append(text)
                    yy.append(y[i])
            self.x = xx
            self.y = yy
        """

        if given_y is not None:
            permperm=[]
            hypohypo=[]
            yy=[]
            for i, label in enumerate(self.y):
                if self.y[i].argmax()==given_y:
                    permperm.append(self.perm[i])
                    hypohypo.append(self.hypo[i])
                    yy.append(self.y[i])
            self.perm = permperm
            self.hypo = hypohypo
            self.y = yy


        #for x in self.syn_data:
        #    self.syn_data[x] = [syn_word for syn_word in self.syn_data[x] if syn_word!=x]

        #self.len_voc = len(self.syn_data)+1
        #super(SynthesizedData_TextLikeSyn_Bert, self).__init__(opt, x, y, syn_data)

    def transform(self, sent, label, text_like_syn, text_like_syn_valid, mask, token_type_ids):
       
        return torch.tensor(sent,dtype=torch.long), torch.tensor(label,dtype=torch.long), torch.tensor(text_like_syn,dtype=torch.long), torch.tensor(text_like_syn_valid,dtype=torch.float), torch.tensor(mask, dtype = torch.long), torch.tensor(token_type_ids, dtype = torch.long)

    def __getitem__(self, index, num_text_like_syn=10):

        encoded = self.tokenizer.encode_plus(self.hypo[index]+" [SEP] "+self.perm[index], None, add_special_tokens = True, max_length = self.seq_max_len, pad_to_max_length = True)

        sent = encoded["input_ids"]
        mask = encoded["attention_mask"]
        token_type_ids = encoded["token_type_ids"]

        text_like_syn=[]
        text_like_syn_valid=[]
        for token in sent:
            text_like_syn_valid.append([])
            text_like_syn.append([token])

        splited_words = self.hypo[index].split()

        for i in range(min(self.seq_max_len-3, len(splited_words))):
            word = splited_words[i]
            if word in self.syn_data:
                text_like_syn[i+1].extend(self.syn_data[word])

        label = self.y[index].argmax()

        for i, x in enumerate(sent):
            text_like_syn_valid[i] = [1 for times in range(len(text_like_syn[i]))]
            
            while(len(text_like_syn[i])<num_text_like_syn):
                text_like_syn[i].append(0)
                text_like_syn_valid[i].append(0)

            assert(len(text_like_syn[i])==num_text_like_syn)
            assert(len(text_like_syn_valid[i])==num_text_like_syn)

        return self.transform(sent, label, text_like_syn, text_like_syn_valid, mask, token_type_ids)

class Snli_SynthesizedData_TextLikeSyn(torch.utils.data.Dataset):
    
    def __init__(self, opt, x_p, x_h, y, syn_data):
        super(Snli_SynthesizedData_TextLikeSyn, self).__init__()
        self.x_p = x_p.copy()
        self.x_h = x_h.copy()
        self.y = y.copy()
        self.syn_data = syn_data.copy()

        max_syn_num = 0

        for x in range(len(self.syn_data)):
            self.syn_data[x] = [syn_word for syn_word in self.syn_data[x] if syn_word!=x]
            max_syn_num =max(max_syn_num, len(self.syn_data[x])+1 )

        print("max syn num:", max_syn_num)
        self.max_syn_num = None
        #self.max_syn_num = max_syn_num

        #self.len_voc = len(self.syn_data)+1

    def transform(self, x_p, x_h, label, x_p_text_like_syn, x_p_text_like_syn_valid, x_h_text_like_syn, x_h_text_like_syn_valid, sent_p_mask, sent_h_mask):
       
        return torch.tensor(x_p,dtype=torch.long), torch.tensor(x_h,dtype=torch.long), torch.tensor(label,dtype=torch.long),torch.tensor(x_p_text_like_syn,dtype=torch.long), torch.tensor(x_p_text_like_syn_valid,dtype=torch.float),torch.tensor(x_h_text_like_syn,dtype=torch.long), torch.tensor(x_h_text_like_syn_valid,dtype=torch.float), torch.tensor(sent_p_mask,dtype=torch.float), torch.tensor(sent_h_mask,dtype=torch.float)

    def __getitem__(self, index, num_text_like_syn=10):

        if self.max_syn_num is not None:
            num_text_like_syn = self.max_syn_num

        sent_p = self.x_p[index]
        sent_h = self.x_h[index]
        label = self.y[index].argmax()

        sent_p_mask = []
        sent_h_mask = []

        x_p_text_like_syn = []
        x_h_text_like_syn = []
        x_p_text_like_syn_valid = []
        x_h_text_like_syn_valid = []

        for x in sent_p:
            if x==0:
                sent_p_mask.append(0)
            else:
                sent_p_mask.append(1)

            x_p_text_like_syn_valid.append([])
            if x<len(self.syn_data):
                x_p_text_like_syn.append( self.syn_data[x].copy() )
            else:
                x_p_text_like_syn.append([])

        for x in sent_h:
            if x==0:
                sent_h_mask.append(0)
            else:
                sent_h_mask.append(1)

            x_h_text_like_syn_valid.append([])
            if x<len(self.syn_data):
                x_h_text_like_syn.append( self.syn_data[x].copy() )
            else:
                x_h_text_like_syn.append([])

        for i, x in enumerate(sent_p):
            temp = x_p_text_like_syn[i]
            len_temp=len(temp)
            if len_temp==0:
                x_p_text_like_syn_valid[i] = [1]
                x_p_text_like_syn_valid[i].extend([0 for times in range(num_text_like_syn-1)])
                x_p_text_like_syn[i] = [x]
                x_p_text_like_syn[i].extend([0 for times in range(num_text_like_syn-1)])
            elif len_temp>=num_text_like_syn-1:
                temp = random.sample(temp, num_text_like_syn-1)
                temp.append(x)
                x_p_text_like_syn[i]=temp
                x_p_text_like_syn_valid[i] = [1 for times in range(num_text_like_syn)]
                assert(len(x_p_text_like_syn[i])==num_text_like_syn)
            else:
                temp.append(x)
                x_p_text_like_syn_valid[i] = [1 for times in range(len(temp))]
                while(len(temp)<num_text_like_syn):
                    temp.append(0)
                    x_p_text_like_syn_valid[i].append(0)
                x_p_text_like_syn[i]=temp
                assert(len(x_p_text_like_syn[i])==num_text_like_syn)

            #if x_p_text_like_syn_valid[i] == [1 for times in range(num_text_like_syn)]:
            #    print(x_p_text_like_syn[i])
            #    print(x_p_text_like_syn_valid[i])

        for i, x in enumerate(sent_h):
            temp = x_h_text_like_syn[i]
            len_temp=len(temp)
            if len_temp==0:
                x_h_text_like_syn_valid[i] = [1]
                x_h_text_like_syn_valid[i].extend([0 for times in range(num_text_like_syn-1)])
                x_h_text_like_syn[i] = [x]
                x_h_text_like_syn[i].extend([0 for times in range(num_text_like_syn-1)])
            elif len_temp>=num_text_like_syn-1:
                temp = random.sample(temp, num_text_like_syn-1)
                temp.append(x)
                x_h_text_like_syn[i]=temp
                x_h_text_like_syn_valid[i] = [1 for times in range(num_text_like_syn)]
                assert(len(x_h_text_like_syn[i])==num_text_like_syn)
            else:
                temp.append(x)
                x_h_text_like_syn_valid[i] = [1 for times in range(len(temp))]
                while(len(temp)<num_text_like_syn):
                    temp.append(0)
                    x_h_text_like_syn_valid[i].append(0)
                x_h_text_like_syn[i]=temp
                assert(len(x_h_text_like_syn[i])==num_text_like_syn)

        return self.transform(sent_p, sent_h, label, x_p_text_like_syn, x_p_text_like_syn_valid, x_h_text_like_syn, x_h_text_like_syn_valid, sent_p_mask, sent_h_mask)

    def __len__(self):
        return len(self.x_p)

class SynonymData(torch.utils.data.Dataset):

    def __init__(self,roots,labels, opt=None):
        super(SynonymData, self).__init__()
        self.roots = roots
        self.negative_src = list(roots.copy())
        random.shuffle(self.negative_src)

        self.labels = labels

        self.len_roots = len(self.roots)

        self.negative_len = 100
        self.negative_sample_idx_list = [ random.randint(0,self.len_roots-1) for x in range(len(roots))]


    def transform(self, root, label, negative_sample_idx_list_start):
    
        root=np.array([root])
        negative = np.zeros(self.negative_len)

        syn_set = set(list(label))
        start = self.negative_sample_idx_list[negative_sample_idx_list_start]

        count = 0
        while (count<self.negative_len):
            if self.negative_src[start] not in syn_set:
                negative[count] = self.negative_src[start]
                count+=1
            start = (start+1) % self.len_roots
        self.negative_sample_idx_list[negative_sample_idx_list_start] = start
        
        #out_root = self.totensor(root).to(torch.long)
        #out_label = self.totensor(label).to(torch.long)
        #out_negative = self.totensor(negative).to(torch.long)

        out_root= torch.from_numpy(root).to(torch.long)
        out_label= torch.from_numpy(label).to(torch.long)
        out_negative = torch.from_numpy(negative).to(torch.long)

        
        return out_root, out_label, out_negative

    def __getitem__(self, index):
        return self.transform(self.roots[index],self.labels[index], index)

    def __len__(self):
        return self.len_roots


class BucketIterator_synonyms(object):
    def __init__(self,roots,labels, opt=None,shuffle=True):
        self.shuffle=shuffle
        self.roots = roots
        self.negative_src = list(roots.copy())
        random.shuffle(self.negative_src)

        self.labels = labels
        self.batch_size=opt.batch_size

        self.len_roots = len(self.roots)

        self.negative_len = 100
        self.negative_sample_idx_list = [ random.randint(0,self.len_roots-1) for x in range(len(roots))]
        

    def transform(self,roots,labels, negative_sample_idx_list_start):

        batch_negatives = np.zeros((self.batch_size, self.negative_len))

        for i in range(self.batch_size):
            syn_set = set(list(labels[i]))
            start = self.negative_sample_idx_list[negative_sample_idx_list_start+i]

            count = 0
            while (count<self.negative_len):
                if self.negative_src[start] not in syn_set:
                    batch_negatives[i,count] = self.negative_src[start]
                    count+=1
                start = (start+1) % self.len_roots
            self.negative_sample_idx_list[negative_sample_idx_list_start+i] = start
        
        if torch.cuda.is_available():
            batch_roots= torch.LongTensor(roots).cuda()
            batch_labels= torch.LongTensor(labels).cuda()
            batch_negatives = torch.LongTensor(batch_negatives).cuda()
        else:
            batch_roots= torch.LongTensor(roots)
            batch_labels= torch.LongTensor(labels)
            batch_negatives = torch.LongTensor(batch_negatives)

        return DottableDict({"roots":batch_roots,"labels":batch_labels,"negatives":batch_negatives})

    def __iter__(self):
        if self.shuffle:
            seed = random.randint(0,100)
            self.roots=list(self.roots)
            random.seed(seed)
            random.shuffle(self.roots)
            self.roots=np.array(self.roots)

            self.labels=list(self.labels)
            random.seed(seed)
            random.shuffle(self.labels)
            self.labels=np.array(self.labels)

            random.seed(seed)
            random.shuffle(self.negative_sample_idx_list)
            

        batch_nums = int(len(self.roots)/self.batch_size)
        for  i in range(batch_nums):
            start=i*self.batch_size
            end=(i+1)*self.batch_size
            yield self.transform(self.roots[start:end],self.labels[start:end], start)
        yield self.transform(self.roots[-1*self.batch_size:],self.labels[-1*self.batch_size:], self.len_roots-self.batch_size)
            
                
#@log_time_delta
def vectors_lookup(vectors,vocab,dim):
    embedding = np.zeros((len(vocab),dim))
    count = 1
    for word in vocab:
        if word in vectors:
            count += 1
            embedding[vocab[word]]= vectors[word]
        else:
            embedding[vocab[word]]= np.random.uniform(-0.5,+0.5,dim)#vectors['[UNKNOW]'] #.tolist()
    print( 'word in embedding',count)
    return embedding

#@log_time_delta
def load_text_vec(alphabet,filename="",embedding_size=-1):
    vectors = {}
    with open(filename,encoding='utf-8') as f:
        for line in tqdm(f):
            items = line.strip().split(' ')
            if len(items) == 2:
                vocab_size, embedding_size= items[0],items[1]
                print( 'embedding_size',embedding_size)
                print( 'vocab_size in pretrained embedding',vocab_size)                
            else:
                word = items[0]
                if word in alphabet:
                    vectors[word] = items[1:]
    print( 'words need to be found ',len(alphabet))
    print( 'words found in wor2vec embedding ',len(vectors.keys()))
    
    if embedding_size==-1:
        embedding_size = len(vectors[list(vectors.keys())[0]])
    return vectors,embedding_size

    
def getDataSet(opt):
    import dataloader
    dataset= dataloader.getDataset(opt)
#    files=[os.path.join(data_dir,data_name)   for data_name in ['train.txt','test.txt','dev.txt']]
    
    return dataset.getFormatedData()
    
    #data_dir = os.path.join(".data/clean",opt.dataset)
    #if not os.path.exists(data_dir):
    #     import dataloader
    #     dataset= dataloader.getDataset(opt)
    #     return dataset.getFormatedData()
    #else:
    #     for root, dirs, files in os.walk(data_dir):
    #         for file in files:
    #             yield os.path.join(root,file)
         
    
#    files=[os.path.join(data_dir,data_name)   for data_name in ['train.txt','test.txt','dev.txt']]
    
import re
def clean(text):
#    text="'tycoon.<br'"
    for token in ["<br/>","<br>","<br"]:
         text = re.sub(token," ",text)
    text = re.sub("[\s+\.\!\/_,$%^*()\(\)<>+\"\[\]\-\?;:\'{}`]+|[+——！，。？、~@#￥%……&*（）]+", " ",text)

#    print("%s $$$$$ %s" %(pre,text))     

    return text.lower().split()
#@log_time_delta
def get_clean_datas(opt):
    pickle_filename = "temp/"+opt.dataset+".data"
    if not os.path.exists(pickle_filename) or opt.debug: 
        datas = [] 
        for filename in getDataSet(opt):
            df = pd.read_csv(filename,header = None,sep="\t",names=["text","label"]).fillna('0')
    
        #        df["text"]= df["text"].apply(clean).str.lower().str.split() #replace("[\",:#]"," ")
            df["text"]= df["text"].apply(clean)
            datas.append(df)
        if  opt.debug:
            if not os.path.exists("temp"):
                os.mkdir("temp")
            pickle.dump(datas,open(pickle_filename,"wb"))
        return datas
    else:
        print("load cache for data")
        return pickle.load(open(pickle_filename,"rb"))

def load_vocab_from_bert(bert_base):
    
    
    bert_vocab_dir = os.path.join(bert_base,"vocab.txt")
    alphabet = Alphabet(start_feature_id = 0,alphabet_type="bert")

    from pytorch_pretrained_bert import BertTokenizer

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(bert_vocab_dir)
    for index,word in tokenizer.ids_to_tokens.items():
        alphabet.add(word)
    return alphabet,tokenizer
        

def load_vocab_for_bert_selfmade():
    
    alphabet = Alphabet(start_feature_id = 0,alphabet_type="bert")

    from transformers import BertTokenizer
    tokenizer_class= BertTokenizer
    tokenizer = tokenizer_class.from_pretrained('bert-base-uncased')

    for index,word in tokenizer.ids_to_tokens.items():
        alphabet.add(word)
    return alphabet,tokenizer

def process_with_bert(text,tokenizer,max_seq_len) :
    tokens =tokenizer.convert_tokens_to_ids(  tokenizer.tokenize(" ".join(text[:max_seq_len])))
    
    return tokens[:max_seq_len] + [0] *int(max_seq_len-len(tokens))

def process_with_bert_selfmade(text,tokenizer,max_seq_len) :
    tokens =tokenizer.encode(  text[:max_seq_len])

    return tokens[:min(len(tokens),max_seq_len)] + [0] *int(max_seq_len-len(tokens))

def snli_make_synthesized_iter(opt):
    dataset=opt.dataset
    opt.label_size = 3
    train_perms, train_hypos, train_labels, dev_perms, dev_hypos, dev_labels, test_perms, test_hypos, test_labels = split_snli_files(opt)
    tokenizer_file = os.path.join(opt.work_path, "temp/snli_tokenizer_with_ceritified_syndata.pickle")

    if opt.synonyms_from_file:
        filename= opt.snli_synonyms_file_path
        f=open(filename,'rb')
        saved=pickle.load(f)
        f.close()
        syn_data = saved["syn_data"]
        opt.embeddings=saved['embeddings']
        opt.vocab_size=saved['vocab_size']
        x_p_train=saved['x_p_train']
        x_h_train=saved['x_h_train']
        y_train=saved['y_train']
        x_p_test=saved['x_p_test']
        x_h_test=saved['x_h_test']
        y_test=saved['y_test']
        x_p_dev=saved['x_p_dev']
        x_h_dev=saved['x_h_dev']
        y_dev=saved['y_dev']

        tokenizer = get_tokenizer(opt)
        dictori=tokenizer.index_word

        
        f=open(tokenizer_file,'rb')
        tokenizer=pickle.load(f)
        f.close()
        update_tokenizer(dataset, tokenizer)
        dictnew=tokenizer.index_word

        print("Check the tokenizer.")
        for key in dictori:
            assert(dictori[key]==dictnew[key])

    else:
        tokenizer = get_tokenizer(opt)
        print("len of tokenizer before updata.", len(tokenizer.index_word))
        print("Preparing synonyms.")

        syn_texts_for_tokenizer = []
        syn_dict = {}

        import json
        with open(opt.certified_neighbors_file_path) as f:
            certified_neighbors = json.load(f)

        syn_data = [[] for i in range(1+len(tokenizer.index_word))]
        for index in tokenizer.index_word:
            if index % 100 == 0:
                print(index)
            word = tokenizer.index_word[index]
            #syn_text = " ".join(generate_synonym_list_from_word(word))
            syn_text = " ".join(generate_synonym_list_by_dict(certified_neighbors, word))
            syn_texts_for_tokenizer.append(syn_text)
            syn_dict[index]=syn_text

        # update tokenizer
        print("Fit on syn texts.")
        dictori=tokenizer.index_word
        tokenizer.fit_on_texts(syn_texts_for_tokenizer, freq_count=0) # to keep the original index order of the tokenizer
        dictnew=tokenizer.index_word
        print("Check the tokenizer.")
        for key in dictori:
            assert(dictori[key]==dictnew[key])

        update_tokenizer(dataset, tokenizer)
        f=open(tokenizer_file,'wb')
        pickle.dump(tokenizer, f)
        f.close()

        print("len of tokenizer after updata.", len(tokenizer.index_word))
        assert(len(tokenizer.index_word)<config.num_words[dataset])

        # Tokenize syn data
        print("Tokenize syn data.")
        for key in syn_dict:
            temp = tokenizer.texts_to_sequences([syn_dict[key]])
            syn_data[key] = temp[0]

        # make embd according to the updated tokenizer
        embedding_dim=opt.embedding_dim
        pretrained = opt.embedding_file_path
        get_embedding_index(pretrained, embedding_dim)
        num_words = config.num_words[dataset]
        opt.vocab_size= num_words
        embedding_matrix = get_embedding_matrix(opt, dataset, num_words, embedding_dim)
        opt.embeddings = torch.FloatTensor(embedding_matrix)

        # Tokenize the training data
        print("Tokenize training data.")
        x_p_train = text_process_for_single(tokenizer, train_perms, opt.dataset)
        x_h_train = text_process_for_single(tokenizer, train_hypos, opt.dataset)
        y_train = label_process_for_single(tokenizer, train_labels, opt.dataset)

        x_p_dev = text_process_for_single(tokenizer, dev_perms, opt.dataset)
        x_h_dev = text_process_for_single(tokenizer, dev_hypos, opt.dataset)
        y_dev = label_process_for_single(tokenizer, dev_labels, opt.dataset)

        x_p_test = text_process_for_single(tokenizer, test_perms, opt.dataset)
        x_h_test = text_process_for_single(tokenizer, test_hypos, opt.dataset)
        y_test = label_process_for_single(tokenizer, test_labels, opt.dataset)
 
        filename= opt.snli_synonyms_file_path
        f=open(filename,'wb')
        saved={}
        saved['syn_data']=syn_data
        saved['embeddings']=opt.embeddings
        saved['vocab_size']=opt.vocab_size
        saved['x_p_train']=x_p_train
        saved['x_h_train']=x_h_train
        saved['y_train']=y_train
        saved['x_p_dev']=x_p_dev
        saved['x_h_dev']=x_h_dev
        saved['y_dev']=y_dev
        saved['x_p_test']=x_p_test
        saved['x_h_test']=x_h_test
        saved['y_test']=y_test
        pickle.dump(saved,f)
        f.close()

    vocab_freq = torch.zeros(opt.embeddings.shape[0]).to(opt.embeddings.dtype)
    for index in tokenizer.index_word:
        word = tokenizer.index_word[index]
        freq = tokenizer.word_counts[word]
        vocab_freq[index] = freq
    opt.vocab_freq = vocab_freq

    train_data = Snli_SynthesizedData_TextLikeSyn(opt, x_p_train, x_h_train, y_train, syn_data)
    train_loader = torch.utils.data.DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=4)

    dev_data = Snli_SynthesizedData_TextLikeSyn(opt, x_p_dev, x_h_dev, y_dev, syn_data)
    dev_loader = torch.utils.data.DataLoader(dev_data, opt.test_batch_size, shuffle=False, num_workers=4)

    test_data = Snli_SynthesizedData_TextLikeSyn(opt, x_p_test, x_h_test, y_test, syn_data)
    test_loader = torch.utils.data.DataLoader(test_data, opt.test_batch_size, shuffle=False, num_workers=4)

    return train_loader, dev_loader, test_loader, syn_data


def imdb_make_synthesized_iter(opt):
    dataset=opt.dataset
    opt.label_size = 2
    train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = split_imdb_files(opt)
    tokenizer_file = os.path.join(opt.work_path, "temp/imdb_tokenizer_with_ceritified_syndata.pickle")

    if opt.synonyms_from_file:
        filename= opt.imdb_synonyms_file_path
        f=open(filename,'rb')
        saved=pickle.load(f)
        f.close()
        syn_data = saved["syn_data"]
        opt.embeddings=saved['embeddings']
        opt.vocab_size=saved['vocab_size']
        x_train=saved['x_train']
        x_test=saved['x_test']
        x_dev=saved['x_dev']
        y_train=saved['y_train']
        y_test=saved['y_test']
        y_dev=saved['y_dev']

        tokenizer = get_tokenizer(opt)
        dictori=tokenizer.index_word

        f=open(tokenizer_file,'rb')
        tokenizer=pickle.load(f)
        f.close()
        update_tokenizer(dataset, tokenizer)
        dictnew=tokenizer.index_word

        print("Check the tokenizer.")
        for key in dictori:
            assert(dictori[key]==dictnew[key])

    else:
        tokenizer = get_tokenizer(opt)
        print("len of tokenizer before updata.", len(tokenizer.index_word))
        print("Preparing synonyms.")

        syn_texts_for_tokenizer = []
        syn_dict = {}

        import json
        with open(opt.certified_neighbors_file_path) as f:
            certified_neighbors = json.load(f)

        syn_data = [[] for i in range(1+len(tokenizer.index_word))]
        for index in tokenizer.index_word:
            if index % 100 == 0:
                print(index)
            word = tokenizer.index_word[index]
            #syn_text = " ".join(generate_synonym_list_from_word(word))
            syn_text = " ".join(generate_synonym_list_by_dict(certified_neighbors, word))
            syn_texts_for_tokenizer.append(syn_text)
            syn_dict[index]=syn_text

        # update tokenizer
        print("Fit on syn texts.")
        tokenizer.fit_on_texts(syn_texts_for_tokenizer, freq_count=0) # to keep the original index order of the tokenizer
        update_tokenizer(dataset, tokenizer)
        f=open(tokenizer_file,'wb')
        pickle.dump(tokenizer, f)
        f.close()

        print("len of tokenizer after updata.", len(tokenizer.index_word))
        assert(len(tokenizer.index_word)<config.num_words[dataset])

        # Tokenize syn data
        print("Tokenize syn data.")
        for key in syn_dict:
            temp = tokenizer.texts_to_sequences([syn_dict[key]])
            syn_data[key] = temp[0]

        # make embd according to the updated tokenizer
        embedding_dim=opt.embedding_dim
        pretrained = opt.embedding_file_path
        get_embedding_index(pretrained, embedding_dim)
        num_words = config.num_words[dataset]
        opt.vocab_size= num_words
        embedding_matrix = get_embedding_matrix(opt, dataset, num_words, embedding_dim)
        opt.embeddings = torch.FloatTensor(embedding_matrix)

        # Tokenize the training data
        print("Tokenize training data.")
        
        x_train = text_process_for_single(tokenizer, train_texts, opt.dataset)
        y_train = label_process_for_single(tokenizer, train_labels, opt.dataset)

        x_dev = text_process_for_single(tokenizer, dev_texts, opt.dataset)
        y_dev = label_process_for_single(tokenizer, dev_labels, opt.dataset)

        x_test = text_process_for_single(tokenizer, test_texts, opt.dataset)
        y_test = label_process_for_single(tokenizer, test_labels, opt.dataset)
 
        filename= opt.imdb_synonyms_file_path
        f=open(filename,'wb')
        saved={}
        saved['syn_data']=syn_data
        saved['embeddings']=opt.embeddings
        saved['vocab_size']=opt.vocab_size
        saved['x_train']=x_train
        saved['x_test']=x_test
        saved['x_dev']=x_dev
        saved['y_train']=y_train
        saved['y_test']=y_test
        saved['y_dev']=y_dev
        pickle.dump(saved,f)
        f.close()

    vocab_freq = torch.zeros(opt.embeddings.shape[0]).to(opt.embeddings.dtype)
    for index in tokenizer.index_word:
        word = tokenizer.index_word[index]
        freq = tokenizer.word_counts[word]
        vocab_freq[index] = freq
    opt.vocab_freq = vocab_freq

    train_data = SynthesizedData_TextLikeSyn(opt, x_train, y_train, syn_data)
    train_loader = torch.utils.data.DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=8)

    # use training data as dev
    dev_data = SynthesizedData_TextLikeSyn(opt, x_dev, y_dev, syn_data)
    dev_loader = torch.utils.data.DataLoader(dev_data, opt.test_batch_size, shuffle=False, num_workers=4)

    test_data = SynthesizedData_TextLikeSyn(opt, x_test, y_test, syn_data)
    test_loader = torch.utils.data.DataLoader(test_data, opt.test_batch_size, shuffle=False, num_workers=4)
    return train_loader, dev_loader, test_loader, syn_data
    
def save_word_vectors_for_show(opt):

    assert(opt.synonyms_from_file)
    filename= opt.synonyms_file_path
    f=open(filename,'rb')
    saved=pickle.load(f)
    f.close()
    syn_data = saved["syn_data"]
    opt.embeddings=saved['embeddings']
    f.close()

    for_save = []

    for i, syn in enumerate(syn_data):
        if len(for_save) == 100:
            break
        if len(syn) ==5:
            for_save.append((opt.embeddings[i].numpy(), [opt.embeddings[x].numpy() for x in syn]))
        
    f=open('word_vectors_for_show_5.pickle','wb')
    pickle.dump(for_save, f)
    f.close()

    print("Done save word vectors for show.")

    return
    

def imdb_make_synthesized_iter_for_bert(opt):
    dataset=opt.dataset
    if dataset == 'imdb':
        opt.label_size = 2
        train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = split_imdb_files(opt)
    elif dataset == 'snli':
        opt.label_size = 2
        train_perms, train_hypos, train_labels, dev_perms, dev_hypos, dev_labels, test_perms, test_hypos, test_labels = split_snli_files(opt)
    elif dataset == 'agnews':
        # opt.label_size = ?
        train_texts, train_labels, test_texts, test_labels = split_agnews_files()
    elif dataset == 'yahoo':
        # opt.label_size = ?
        train_texts, train_labels, test_texts, test_labels = split_yahoo_files()
            

    from modified_bert_tokenizer import ModifiedBertTokenizer
    #import transformers
    #tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    tokenizer = ModifiedBertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    if opt.synonyms_from_file:
        filename= opt.imdb_bert_synonyms_file_path
        f=open(filename,'rb')
        saved=pickle.load(f)
        f.close()
        syn_data = saved["syn_data"]
        #x_train=saved['x_train']
        #x_test=saved['x_test']
        #x_dev=saved['x_dev']
        y_train=saved['y_train']
        y_test=saved['y_test']
        y_dev=saved['y_dev']

    else:
        #tokenizer = get_tokenizer(opt)
        #print("len of tokenizer before updata.", len(tokenizer.index_word))
        print("Preparing synonyms.")

        syn_dict = get_syn_dict(opt)
        syn_data = {} # key is textual word

        # Tokenize syn data
        print("Tokenize syn data.")
        for key in syn_dict:
            if len(syn_dict[key])!=0:
                temp = tokenizer.encode_plus(syn_dict[key], None, add_special_tokens=False, pad_to_max_length=False)['input_ids']

                token_of_key = tokenizer.encode_plus(key, None, add_special_tokens=False, pad_to_max_length=False)["input_ids"][0]

                syn_data[key] = temp

                #if not token_of_key in syn_data:
                #    syn_data[token_of_key] = temp
                #else:
                #    syn_data[token_of_key].append(temp)

        # Tokenize the training data
        print("Tokenize training data.")
        #x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)
        
        #x_train = text_process_for_single_bert(tokenizer, train_texts, opt.dataset)
        y_train = label_process_for_single(tokenizer, train_labels, opt.dataset)

        #x_dev = text_process_for_single_bert(tokenizer, dev_texts, opt.dataset)
        y_dev = label_process_for_single(tokenizer, dev_labels, opt.dataset)

        #x_test = text_process_for_single_bert(tokenizer, test_texts, opt.dataset)
        y_test = label_process_for_single(tokenizer, test_labels, opt.dataset)
 

        filename= opt.imdb_bert_synonyms_file_path
        f=open(filename,'wb')
        saved={}
        saved['syn_data']=syn_data
        #saved['x_train']=x_train
        #saved['x_test']=x_test
        #saved['x_dev']=x_dev
        saved['y_train']=y_train
        saved['y_test']=y_test
        saved['y_dev']=y_dev
        pickle.dump(saved,f)
        f.close()

    from PWWS.config import config
    seq_max_len = config.word_max_len[opt.dataset]

    train_data = SynthesizedData_TextLikeSyn_Bert(opt, train_texts, y_train, syn_data, seq_max_len, tokenizer, given_y=None)
    train_data.__getitem__(0)
    train_loader = torch.utils.data.DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=4)

    # use training data as dev
    dev_data = SynthesizedData_TextLikeSyn_Bert(opt, dev_texts, y_dev, syn_data, seq_max_len, tokenizer)
    dev_loader = torch.utils.data.DataLoader(dev_data, opt.test_batch_size, shuffle=False, num_workers=4)

    from from_certified.attack_surface import WordSubstitutionAttackSurface, LMConstrainedAttackSurface
    attack_surface = LMConstrainedAttackSurface.from_files(opt.certified_neighbors_file_path, opt.imdb_lm_file_path)

    test_data = SynthesizedData_TextLikeSyn_Bert(opt, test_texts, y_test, syn_data, seq_max_len, tokenizer, attack_surface=attack_surface)
    test_loader = torch.utils.data.DataLoader(test_data, opt.test_batch_size, shuffle=False, num_workers=4)
    return train_loader, dev_loader, test_loader, syn_data


def imdb_bert_make_synthesized_iter_giveny(opt):
    dataset=opt.dataset
    opt.label_size = 2
    train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = split_imdb_files(opt)

    from modified_bert_tokenizer import ModifiedBertTokenizer
    #import transformers
    #tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    tokenizer = ModifiedBertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    if opt.synonyms_from_file:
        filename= opt.imdb_bert_synonyms_file_path
        f=open(filename,'rb')
        saved=pickle.load(f)
        f.close()
        syn_data = saved["syn_data"]
        #x_train=saved['x_train']
        #x_test=saved['x_test']
        #x_dev=saved['x_dev']
        y_train=saved['y_train']
        y_test=saved['y_test']
        y_dev=saved['y_dev']

    else:
        #tokenizer = get_tokenizer(opt)
        #print("len of tokenizer before updata.", len(tokenizer.index_word))
        print("Preparing synonyms.")

        syn_dict = get_syn_dict(opt)
        syn_data = {} # key is textual word

        # Tokenize syn data
        print("Tokenize syn data.")
        for key in syn_dict:
            if len(syn_dict[key])!=0:
                temp = tokenizer.encode_plus(syn_dict[key], None, add_special_tokens=False, pad_to_max_length=False)['input_ids']

                token_of_key = tokenizer.encode_plus(key, None, add_special_tokens=False, pad_to_max_length=False)["input_ids"][0]

                syn_data[key] = temp

                #if not token_of_key in syn_data:
                #    syn_data[token_of_key] = temp
                #else:
                #    syn_data[token_of_key].append(temp)

        # Tokenize the training data
        print("Tokenize training data.")
        #x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)
        
        #x_train = text_process_for_single_bert(tokenizer, train_texts, opt.dataset)
        y_train = label_process_for_single(tokenizer, train_labels, opt.dataset)

        #x_dev = text_process_for_single_bert(tokenizer, dev_texts, opt.dataset)
        y_dev = label_process_for_single(tokenizer, dev_labels, opt.dataset)

        #x_test = text_process_for_single_bert(tokenizer, test_texts, opt.dataset)
        y_test = label_process_for_single(tokenizer, test_labels, opt.dataset)
 

        filename= opt.imdb_bert_synonyms_file_path
        f=open(filename,'wb')
        saved={}
        saved['syn_data']=syn_data
        #saved['x_train']=x_train
        #saved['x_test']=x_test
        #saved['x_dev']=x_dev
        saved['y_train']=y_train
        saved['y_test']=y_test
        saved['y_dev']=y_dev
        pickle.dump(saved,f)
        f.close()

    from PWWS.config import config
    seq_max_len = config.word_max_len[dataset]

    opt.n_training_set = len(train_texts)

    train_data_y0 = SynthesizedData_TextLikeSyn_Bert(opt, train_texts, y_train, syn_data, seq_max_len, tokenizer, given_y=0)
    train_data_y0.__getitem__(0)
    train_loader_y0 = torch.utils.data.DataLoader(train_data_y0, opt.batch_size//2, shuffle=True, num_workers=16)

    train_data_y1 = SynthesizedData_TextLikeSyn_Bert(opt, train_texts, y_train, syn_data, seq_max_len, tokenizer, given_y=1)
    train_loader_y1 = torch.utils.data.DataLoader(train_data_y1, opt.batch_size//2, shuffle=True, num_workers=16)

    # use training data as dev
    dev_data = SynthesizedData_TextLikeSyn_Bert(opt, dev_texts, y_dev, syn_data, seq_max_len, tokenizer)
    dev_loader = torch.utils.data.DataLoader(dev_data, opt.test_batch_size, shuffle=False, num_workers=16)

    from from_certified.attack_surface import WordSubstitutionAttackSurface, LMConstrainedAttackSurface
    attack_surface = LMConstrainedAttackSurface.from_files(opt.certified_neighbors_file_path, opt.imdb_lm_file_path)

    test_data = SynthesizedData_TextLikeSyn_Bert(opt, test_texts, y_test, syn_data, seq_max_len, tokenizer, attack_surface=attack_surface)
    test_loader = torch.utils.data.DataLoader(test_data, opt.test_batch_size, shuffle=False, num_workers=16)
    return train_loader_y0, train_loader_y1, dev_loader, test_loader, syn_data


def snli_make_synthesized_iter_for_bert_giveny(opt):
    dataset=opt.dataset
    if dataset == 'snli':
        opt.label_size = 3
        train_perms, train_hypos, train_labels, dev_perms, dev_hypos, dev_labels, test_perms, test_hypos, test_labels = split_snli_files(opt)
      
    from modified_bert_tokenizer import ModifiedBertTokenizer
    #import transformers
    #tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    tokenizer = ModifiedBertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    if opt.synonyms_from_file:
        filename= opt.snli_bert_synonyms_file_path
        f=open(filename,'rb')
        saved=pickle.load(f)
        f.close()
        syn_data = saved["syn_data"]
        #x_p_train=saved['x_p_train']
        #x_h_train=saved['x_h_train']
        y_train=saved['y_train']
        #x_p_test=saved['x_p_test']
        #x_h_test=saved['x_h_test']
        y_test=saved['y_test']
        #x_p_dev=saved['x_p_dev']
        #x_h_dev=saved['x_h_dev']
        y_dev=saved['y_dev']
    else:
        #tokenizer = get_tokenizer(opt)
        #print("len of tokenizer before updata.", len(tokenizer.index_word))
        print("Preparing synonyms.")

        syn_dict = get_syn_dict(opt)
        syn_data = {} # key is textual word

        # Tokenize syn data
        print("Tokenize syn data.")
        for key in syn_dict:
            if len(syn_dict[key])!=0:
                temp = tokenizer.encode_plus(syn_dict[key], None, add_special_tokens=False, pad_to_max_length=False)['input_ids']

                token_of_key = tokenizer.encode_plus(key, None, add_special_tokens=False, pad_to_max_length=False)["input_ids"][0]

                syn_data[key] = temp

                #if not token_of_key in syn_data:
                #    syn_data[token_of_key] = temp
                #else:
                #    syn_data[token_of_key].append(temp)

        # Tokenize the training data
        print("Tokenize training data.")
        #x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)
        
        #x_train = text_process_for_single_bert(tokenizer, train_texts, opt.dataset)
        y_train = label_process_for_single(tokenizer, train_labels, opt.dataset)

        #x_dev = text_process_for_single_bert(tokenizer, dev_texts, opt.dataset)
        y_dev = label_process_for_single(tokenizer, dev_labels, opt.dataset)

        #x_test = text_process_for_single_bert(tokenizer, test_texts, opt.dataset)
        y_test = label_process_for_single(tokenizer, test_labels, opt.dataset)
 

        filename= opt.snli_bert_synonyms_file_path
        f=open(filename,'wb')
        saved={}
        saved['syn_data']=syn_data
        #saved['x_train']=x_train
        #saved['x_test']=x_test
        #saved['x_dev']=x_dev
        saved['y_train']=y_train
        saved['y_test']=y_test
        saved['y_dev']=y_dev
        pickle.dump(saved,f)
        f.close()



    from PWWS.config import config
    seq_max_len = config.word_max_len[opt.dataset]

    train_data_y0 = SNLI_SynthesizedData_TextLikeSyn_Bert(opt, train_perms, train_hypos, y_train, syn_data, seq_max_len, tokenizer, given_y=0)
    train_loader_y0 = torch.utils.data.DataLoader(train_data_y0, opt.batch_size//opt.label_size, shuffle=True, num_workers=4)

    train_data_y0.__getitem__(0)


    train_data_y1 = SNLI_SynthesizedData_TextLikeSyn_Bert(opt, train_perms, train_hypos, y_train, syn_data, seq_max_len, tokenizer, given_y=1)
    train_loader_y1 = torch.utils.data.DataLoader(train_data_y1, opt.batch_size//opt.label_size, shuffle=True, num_workers=4)

    train_data_y2 = SNLI_SynthesizedData_TextLikeSyn_Bert(opt, train_perms, train_hypos, y_train, syn_data, seq_max_len, tokenizer, given_y=2)
    train_loader_y2 = torch.utils.data.DataLoader(train_data_y2, opt.batch_size//opt.label_size, shuffle=True, num_workers=4)

    # use training data as dev
    dev_data = SNLI_SynthesizedData_TextLikeSyn_Bert(opt, dev_perms, dev_hypos, y_dev, syn_data, seq_max_len, tokenizer)
    dev_loader = torch.utils.data.DataLoader(dev_data, opt.test_batch_size, shuffle=False, num_workers=4)

    from from_certified.attack_surface import WordSubstitutionAttackSurface, LMConstrainedAttackSurface
    attack_surface = LMConstrainedAttackSurface.from_files(opt.certified_neighbors_file_path, opt.imdb_lm_file_path)

    test_data = SNLI_SynthesizedData_TextLikeSyn_Bert(opt, test_perms, test_hypos, y_test, syn_data, seq_max_len, tokenizer, attack_surface=attack_surface)
    test_loader = torch.utils.data.DataLoader(test_data, opt.test_batch_size, shuffle=False, num_workers=4)
    return train_loader_y0, train_loader_y1, train_loader_y2, dev_loader, test_loader, syn_data


def load_synonyms_in_vocab(opt, max_synonym_num = 1000):
    if opt.synonyms_from_file:
        filename= opt.synonyms_file_path
        f=open(filename,'rb')
        saved=pickle.load(f)
        f.close()
        roots = saved["roots"]
        labels = saved["labels"]

        #all_words_set = set()
        #tokenizer = get_tokenizer(opt)
        #for index in tokenizer.index_word:
        #    all_words_set.add(index)
        
        #import time
        #t0 = time.time()
        #negatives = []
        #for idx, x in enumerate(labels):
        #    negatives.append( np.array(list(all_words_set-set(list(x))))  )
        #    if idx %100 ==0:
        #        print(time.time()-t0, idx)
        #        t0=time.time()
        #print("Done.")

        #f=open(filename,'wb')
        #saved['negatives']=negatives
        #pickle.dump(saved,f)
        #f.close()

    else:
        tokenizer = get_tokenizer(opt)
        print("Preparing synonyms.")

        data = [[] for i in range(len(tokenizer.index_word))]

        for index in tokenizer.index_word:
            if index % 100 == 0:
                print(index)
            #if index>100:
            #    break

            word = tokenizer.index_word[index]
            synonym_list_ori = generate_synonym_list_from_word(word)

            synonym_list = []

            if len(synonym_list_ori) != 0:
                for synonym in synonym_list_ori:
                    temp = tokenizer.texts_to_sequences([synonym])
                    if temp!= [[]]:
                        synonym_list.append(temp[0][0])

            data[index] = synonym_list

        filename= opt.synonyms_file_path
        f=open(filename,'wb')
        saved={}
        saved['data']=data
        pickle.dump(saved,f)
        f.close()
        
    syn_data = SynonymData(roots,labels, opt)
    syn_loader = torch.utils.data.DataLoader(syn_data, opt.syn_batch_size, shuffle=True, num_workers=4)
    #syn_data.__getitem__(0)
    return syn_loader
    #return BucketIterator_synonyms(roots,labels, opt)


if __name__ =="__main__":
    import opts
