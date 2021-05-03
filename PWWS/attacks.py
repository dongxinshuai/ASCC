import itertools
import glob
import json
import numpy as np
import os
import pickle
import random

from nltk import word_tokenize
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from keras.preprocessing import sequence

"""
LOSS_FUNC = nn.BCEWithLogitsLoss()
IMDB_DIR = 'data/aclImdb'
LM_FILE = 'data/lm_scores/imdb_all.txt'
COUNTER_FITTED_FILE = 'data/counter-fitted-vectors.txt'
"""
class AdversarialModel_Snli(object):
    def __init__(self, model, tokenizer, maxlen):
        super(AdversarialModel_Snli, self).__init__()
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.maxlen = maxlen
    def query(self, x_p, x_h, device):
        x_p = self.tokenizer.texts_to_sequences([x_p])
        x_p_mask = [1 for i in range(len(x_p[0]))]+[0 for i in range(self.maxlen-len(x_p[0]))]
        x_p_mask = [x_p_mask]
        x_p_mask = np.array(x_p_mask)
        x_p_mask=torch.from_numpy(x_p_mask).to(device).to(torch.float)

        x_p = sequence.pad_sequences(x_p, maxlen=self.maxlen, padding='post', truncating='post')
        x_p=torch.from_numpy(x_p).to(device).to(torch.long)

        x_h = self.tokenizer.texts_to_sequences([x_h])
        x_h_mask = [1 for i in range(len(x_h[0]))]+[0 for i in range(self.maxlen-len(x_h[0]))]
        x_h_mask = [x_h_mask]
        x_h_mask = np.array(x_h_mask)
        x_h_mask=torch.from_numpy(x_h_mask).to(device).to(torch.float)

        x_h = sequence.pad_sequences(x_h, maxlen=self.maxlen, padding='post', truncating='post')
        x_h=torch.from_numpy(x_h).to(device).to(torch.long)
        #input_vector = np.zeros((1, self.maxlen))
        #input_vector = x
        
        logit = self.model(mode="text_to_logit",x_p=x_p, x_h=x_h, x_p_mask=x_p_mask, x_h_mask=x_h_mask).squeeze(0)

        return logit.detach().cpu().numpy()
        #p = F.softmax(logit).detach().cpu().numpy()
        #return p.squeeze()

class AdversarialModel(object):
    def __init__(self, model, tokenizer, maxlen, test_bert=False):
        super(AdversarialModel, self).__init__()
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.test_bert = test_bert

    def tokenize(self, x):
        if self.test_bert:
            #print(len(x.split()))
            #print(x)
            token = self.tokenizer.encode_plus(x, None, add_special_tokens=True, max_length=self.maxlen, pad_to_max_length=True)

            return token
        else:
            token = self.tokenizer.texts_to_sequences([x])
            token = sequence.pad_sequences(token, maxlen=self.maxlen, padding='post', truncating='post')
            return token

    def query(self, x, device):

        #input_vector = self.tokenizer.texts_to_sequences([x])
        #input_vector = sequence.pad_sequences(input_vector, maxlen=self.maxlen, padding='post', truncating='post')

        if self.test_bert:
            token = self.tokenize(x)

            text = np.array([token['input_ids']])
            text = torch.tensor(text,dtype=torch.long).to(device)
            mask = np.array([token['attention_mask']])
            #print(mask.sum())
            mask = torch.tensor(mask,dtype=torch.long).to(device)
            token_type_ids = np.array([token["token_type_ids"]])
            token_type_ids = torch.tensor(token_type_ids,dtype=torch.long).to(device)

            logit = self.model(mode="text_to_logit", input=text, bert_mask=mask, bert_token_id=token_type_ids).squeeze(0)

        else:
            token = self.tokenize(x)
            input_vector=torch.from_numpy(token).to(device).to(torch.long)
            logit = self.model(mode="text_to_logit",input=input_vector).squeeze(0)

        p = F.softmax(logit).detach().cpu().numpy()
        return p
        

class Adversary(object):
    def __init__(self, attack_surface):
        self.attack_surface = attack_surface

    def run(self, model, dataset, device, opts=None):
        """Run adversary on a dataset.

        Args:
        model: a TextClassificationModel.
        dataset: a TextClassificationDataset.
        device: torch device.
        Returns: pair of
        - list of 0-1 adversarial loss of same length as |dataset|
        - list of list of adversarial examples (each is just a text string)
        """
        raise NotImplementedError




class GeneticAdversary_Snli(Adversary):
    def __init__(self, attack_surface, num_iters=20, pop_size=60, margin_goal=0):
        super(GeneticAdversary_Snli, self).__init__(attack_surface)
        self.num_iters = num_iters
        self.pop_size = pop_size
        self.margin_goal = margin_goal


    def get_margins(self, model_output, gold_labels):
        logits = model_output

        true_class_pred = logits[gold_labels]

        temp = logits.copy()
        temp[gold_labels]=-1e20
        highest_false_pred = temp.max()
        value_margin = true_class_pred - highest_false_pred
        return value_margin

    def perturb(self, x_p, hypo, choices, model, y, device):
        if all(len(c) == 1 for c in choices):
            value_margin = self.get_margins( model.query(x_p, ' '.join(hypo), device), y)
            return hypo, value_margin.item()
        good_idxs = [i for i, c in enumerate(choices) if len(c) > 1]
        idx = random.sample(good_idxs, 1)[0]
        x_h_list = [' '.join(hypo[:idx] + [w_new] + hypo[idx+1:])
                for w_new in choices[idx]]
        querry_list = [model.query(x_p, x_h, device) for x_h in x_h_list]
        best_replacement_idx = None
        worst_margin = float('inf')
        for idx_in_choices, logits in enumerate(querry_list):
            value_margin = self.get_margins(logits, y)
            if best_replacement_idx is None or value_margin.item() < worst_margin:
                best_replacement_idx = idx_in_choices
                worst_margin = value_margin.item()

        cur_words = list(hypo)
        cur_words[idx] = choices[idx][best_replacement_idx]
        return cur_words, worst_margin

    def run(self, model, texts_p, texts_h, labels, device, genetic_test_num, opts=None):
        is_correct = []
        adv_exs = []
        total = 0
        acc = 0

        #texts_p = texts_p[opts.h_test_start:opts.h_test_start+opts.genetic_test_num]
        #texts_h = texts_h[opts.h_test_start:opts.h_test_start+opts.genetic_test_num]
        #labels = labels[opts.h_test_start:opts.h_test_start+opts.genetic_test_num]

        for x_p, x_h, y in zip(texts_p, texts_h, labels):

            words = x_h.split()
            if not self.attack_surface.check_in(words):
                continue

            print(acc, "/", total)
            if total >= genetic_test_num:
                break
            total += 1
            y = np.argmax(y) 
            # First query the example itself
            orig_pred = model.query(x_p, x_h, device)
            if np.argmax(orig_pred) != y :
                print('ORIGINAL PREDICTION WAS WRONG')
                is_correct.append(0)
                adv_exs.append((x_p, x_h))
                continue
            # Now run adversarial search
            x_h_words = x_h.split()
            swaps = self.attack_surface.get_swaps(x_h_words)
            choices = [[w] + cur_swaps for w, cur_swaps in zip(x_h_words, swaps)]
            found = False
            population = [self.perturb(x_p, x_h_words, choices, model, y, device)
                            for i in range(self.pop_size)]
            for g in range(self.num_iters):
                best_idx = min(enumerate(population), key=lambda x: x[1][1])[0]
                #print('Iteration %d: %.6f' % (g, population[best_idx][1]))
                if population[best_idx][1] < self.margin_goal:
                    found = True
                    is_correct.append(0)
                    adv_exs.append(' '.join(population[best_idx][0]))
                    #print('ADVERSARY SUCCESS on ("%s", %d): Found "%s" with margin %.2f' % (x, y, adv_exs[-1], population[best_idx][1]))
                    print('ADVERSARY SUCCESS')
                    break
                new_population = [population[best_idx]]

                margins = np.array([m for c, m in population])
                adv_probs = 1 / (1 + np.exp(margins)) + 1e-4
                # Sigmoid of negative margin, for probabilty of wrong class
                # Add 1e-6 for numerical stability
                sample_probs = adv_probs / np.sum(adv_probs)

                for i in range(1, self.pop_size):
                    parent1 = population[np.random.choice(range(len(population)), p=sample_probs)][0]
                    parent2 = population[np.random.choice(range(len(population)), p=sample_probs)][0]
                    child = [random.sample([w1, w2], 1)[0] for (w1, w2) in zip(parent1, parent2)]
                    child_mut, new_margin = self.perturb(x_p, child, choices, model, y, device)
                    new_population.append((child_mut, new_margin))
                population = new_population
            else:
                is_correct.append(1)
                adv_exs.append([])
                acc+=1
                #print('ADVERSARY FAILURE on ("%s", %d)' % (x, y))
                print('ADVERSARY FAILURE', 'Iteration %d: %.6f' % (g, population[best_idx][1]))
        if total != 0:
            return acc*1.0/total
        else:
            return 0

class GeneticAdversary(Adversary):
    def __init__(self, attack_surface, num_iters=20, pop_size=60, margin_goal=0.5):
        super(GeneticAdversary, self).__init__(attack_surface)
        self.num_iters = num_iters
        self.pop_size = pop_size
        self.margin_goal = margin_goal


    def perturb(self, words, choices, model, y, device):
        if all(len(c) == 1 for c in choices): return words
        good_idxs = [i for i, c in enumerate(choices) if len(c) > 1]
        idx = random.sample(good_idxs, 1)[0]
        x_list = [' '.join(words[:idx] + [w_new] + words[idx+1:])
                for w_new in choices[idx]]
        preds = [model.query(x, device) for x in x_list]
        preds_of_y = [p[y] for p in preds]
        best_idx = min(enumerate(preds_of_y), key=lambda x: x[1])[0]
        cur_words = list(words)
        cur_words[idx] = choices[idx][best_idx]
        return cur_words, preds_of_y[best_idx]

    def run(self, model, texts, labels, device, genetic_test_num, opts=None):
        is_correct = []
        adv_exs = []
        total = 0
        acc = 0

        #texts = texts[opts.h_test_start:opts.h_test_start+opts.genetic_test_num]
        #labels = labels[opts.h_test_start:opts.h_test_start+opts.genetic_test_num]

        for x, y in zip(texts, labels):
            words = x.split()
            if not self.attack_surface.check_in(words):
                continue

            print(acc, "/", total)
            if total >= genetic_test_num:
                break
            total += 1
            y = np.argmax(y) 
            # First query the example itself
            orig_pred = model.query(x, device)
            if np.argmax(orig_pred) != y :
                print('ORIGINAL PREDICTION WAS WRONG')
                is_correct.append(0)
                adv_exs.append(x)
                continue
            # Now run adversarial search
            words = x.split()
            swaps = self.attack_surface.get_swaps(words)
            choices = [[w] + cur_swaps for w, cur_swaps in zip(words, swaps)]
            found = False
            population = [self.perturb(words, choices, model, y, device)
                            for i in range(self.pop_size)]
            for g in range(self.num_iters):
                best_idx = min(enumerate(population), key=lambda x: x[1][1])[0]
                #print('Iteration %d: %.6f' % (g, population[best_idx][1]))
                if population[best_idx][1] < self.margin_goal:
                    found = True
                    is_correct.append(0)
                    adv_exs.append(' '.join(population[best_idx][0]))
                    #print('ADVERSARY SUCCESS on ("%s", %d): Found "%s" with margin %.2f' % (x, y, adv_exs[-1], population[best_idx][1]))
                    print('ADVERSARY SUCCESS')
                    break
                new_population = [population[best_idx]]
                p_y = np.array([m for c, m in population])
                temp = 1-p_y + 1e-8
                sample_probs = (temp) / np.sum(temp) 
                #sample_probs = sample_probs + 1e-8
                for i in range(1, self.pop_size):
                    parent1 = population[np.random.choice(range(len(population)), p=sample_probs)][0]
                    parent2 = population[np.random.choice(range(len(population)), p=sample_probs)][0]
                    child = [random.sample([w1, w2], 1)[0] for (w1, w2) in zip(parent1, parent2)]
                    child_mut, new_p_y = self.perturb(child, choices, model, y, device)
                    new_population.append((child_mut, new_p_y))
                population = new_population
            else:
                is_correct.append(1)
                adv_exs.append([])
                acc+=1
                #print('ADVERSARY FAILURE on ("%s", %d)' % (x, y))
                print('ADVERSARY FAILURE', 'Iteration %d: %.6f' % (g, population[best_idx][1]))
        if total != 0:
            return acc*1.0/total
        else:
            return 0