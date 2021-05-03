import sys
import keras
import spacy
import numpy as np
import tensorflow as tf
import os
from .config import config
from keras import backend as K
from .paraphrase import _compile_perturbed_tokens, PWWS, PWWS_snli
from .word_level_process import text_to_vector
from .char_level_process import doc_process, get_embedding_dict
from .evaluate_word_saliency import evaluate_word_saliency, evaluate_word_saliency_snli
#from keras.backend.tensorflow_backend import set_session
from .unbuffered import Unbuffered

import torch.nn.functional as F
import torch

sys.stdout = Unbuffered(sys.stdout)
nlp = spacy.load('en', tagger=False, entity=False)


class ForwardGradWrapper:
    '''
    Utility class that computes the classification probability of model input and predict its class
    '''

    def __init__(self, model):
        '''
        :param model: Keras model.
            This code makes a bunch of assumptions about the model:
            - Model has single input
            - Embedding is the first layer
            - Model output is a scalar (logistic regression)
        '''
        input_tensor = model.input

        self.model = model
        self.input_tensor = input_tensor
        self.sess = K.get_session()

    def predict_prob(self, input_vector):
        prob = self.model.predict(input_vector).squeeze()
        return prob

    def predict_classes(self, input_vector):
        prediction = self.model.predict(input_vector)
        classes = np.argmax(prediction, axis=1)
        return classes


class ForwardGradWrapper_pytorch_snli:
    '''
    Utility class that computes the classification probability of model input and predict its class
    '''

    def __init__(self, model, device):
        '''
        :param model: Keras model.
            This code makes a bunch of assumptions about the model:
            - Model has single input
            - Embedding is the first layer
            - Model output is a scalar (logistic regression)
        '''
        model.eval()
        self.model=model
        self.device=device
    
    def get_mask(self, tensor):
        #mask = 1- (tensor==0)
        mask = ~(tensor==0)
        mask=mask.to(self.device).to(torch.float)
        return mask

    def predict_prob(self, input_vector_p, input_vector_h):
        input_vector_p=torch.from_numpy(input_vector_p).to(self.device).to(torch.long)
        input_vector_h=torch.from_numpy(input_vector_h).to(self.device).to(torch.long)
        mask_p = self.get_mask(input_vector_p)
        mask_h = self.get_mask(input_vector_h)
        
        logit = self.model(mode="text_to_logit",x_p=input_vector_p, x_h=input_vector_h, x_p_mask=mask_p, x_h_mask=mask_h).squeeze(0)
        return F.softmax(logit).detach().cpu().numpy()

    def predict_classes(self, input_vector_p, input_vector_h):
        input_vector_p=torch.from_numpy(input_vector_p).to(self.device).to(torch.long)
        input_vector_h=torch.from_numpy(input_vector_h).to(self.device).to(torch.long)
        mask_p = self.get_mask(input_vector_p)
        mask_h = self.get_mask(input_vector_h)
        
        logit = self.model(mode="text_to_logit",x_p=input_vector_p, x_h=input_vector_h, x_p_mask=mask_p, x_h_mask=mask_h).squeeze(0)
        logit=logit.detach().cpu().numpy()
        classes = np.argmax(logit, axis=-1)
        return classes

class ForwardGradWrapper_pytorch:
    '''
    Utility class that computes the classification probability of model input and predict its class
    '''

    def __init__(self, model, device):
        '''
        :param model: Keras model.
            This code makes a bunch of assumptions about the model:
            - Model has single input
            - Embedding is the first layer
            - Model output is a scalar (logistic regression)
        '''
        model.eval()
        self.model=model
        self.device=device
        

    def predict_prob(self, input_vector):
        input_vector=torch.from_numpy(input_vector).to(self.device).to(torch.long)
        logit = self.model(mode="text_to_logit",input=input_vector).squeeze(0)
        return F.softmax(logit).detach().cpu().numpy()

    def predict_classes(self, input_vector):
        input_vector=torch.from_numpy(input_vector).to(self.device).to(torch.long)
        logit = self.model(mode="text_to_logit",input=input_vector).squeeze(0)
        logit=logit.detach().cpu().numpy()
        classes = np.argmax(logit, axis=-1)
        return classes


def adversarial_paraphrase(opt, input_text, true_y, grad_guide, tokenizer, dataset, level, verbose=True):
    '''
    Compute a perturbation, greedily choosing the synonym if it causes the most
    significant change in the classification probability after replacement
    :return perturbed_text: generated adversarial examples
    :return perturbed_y: predicted class of perturbed_text
    :return sub_rate: word replacement rate showed in Table 3
    :return change_tuple_list: list of substitute words
    '''

    def halt_condition_fn(perturbed_text):
        '''
        Halt if model output is changed.
        '''
        perturbed_vector = None
        if level == 'word':
            perturbed_vector = text_to_vector(perturbed_text, tokenizer, dataset)
        elif level == 'char':
            max_len = config.char_max_len[dataset]
            perturbed_vector = doc_process(perturbed_text, get_embedding_dict(), dataset).reshape(1, max_len)
        adv_y = grad_guide.predict_classes(input_vector=perturbed_vector)
        if adv_y != true_y:
            return True
        else:
            return False

    def heuristic_fn(text, candidate):
        '''
        Return the difference between the classification probability of the original
        word and the candidate substitute synonym, which is defined in Eq.(4) and Eq.(5).
        '''
        doc = nlp(text)
        origin_vector = None
        perturbed_vector = None
        if level == 'word':
            origin_vector = text_to_vector(text, tokenizer, dataset)

            perturbed_text_list = _compile_perturbed_tokens(doc, [candidate])
            perturbed_text = "" 
            for i, word_str in enumerate(perturbed_text_list):
                if i==0:
                    perturbed_text+=word_str
                else:
                    if word_str[0] in [".", ",", "-", "'", ":", "!", "?", "(", ")", ";", "<", ">"]:
                        perturbed_text+=word_str
                    else:
                        perturbed_text+=(" "+word_str)

            perturbed_doc = nlp(perturbed_text)
            perturbed_vector = text_to_vector(perturbed_doc.text, tokenizer, dataset)
        elif level == 'char':
            max_len = config.char_max_len[dataset]
            origin_vector = doc_process(text, get_embedding_dict(), dataset).reshape(1, max_len)
            perturbed_tokens = _compile_perturbed_tokens(nlp(input_text), [candidate])
            perturbed_text = ' '.join(perturbed_tokens)
            perturbed_vector = doc_process(perturbed_text, get_embedding_dict(), dataset).reshape(1, max_len)

        origin_prob = grad_guide.predict_prob(input_vector=origin_vector)
        perturbed_prob = grad_guide.predict_prob(input_vector=perturbed_vector)
        delta_p = origin_prob[true_y] - perturbed_prob[true_y]

        return delta_p

    doc = nlp(input_text)

    # PWWS
    position_word_list, word_saliency_list = evaluate_word_saliency(doc, grad_guide, tokenizer, true_y, dataset, level)
    perturbed_text, sub_rate, NE_rate, change_tuple_list = PWWS(opt,
                                                                doc,
                                                                true_y,
                                                                dataset,
                                                                word_saliency_list=word_saliency_list,
                                                                heuristic_fn=heuristic_fn,
                                                                halt_condition_fn=halt_condition_fn,
                                                                verbose=verbose)

    # print("perturbed_text after perturb_text:", perturbed_text)
    origin_vector = perturbed_vector = None
    if level == 'word':
        origin_vector = text_to_vector(input_text, tokenizer, dataset)
        perturbed_vector = text_to_vector(perturbed_text, tokenizer, dataset)
    elif level == 'char':
        max_len = config.char_max_len[dataset]
        origin_vector = doc_process(input_text, get_embedding_dict(), dataset).reshape(1, max_len)
        perturbed_vector = doc_process(perturbed_text, get_embedding_dict(), dataset).reshape(1, max_len)
    perturbed_y = grad_guide.predict_classes(input_vector=perturbed_vector)
    if verbose:
        origin_prob = grad_guide.predict_prob(input_vector=origin_vector)
        perturbed_prob = grad_guide.predict_prob(input_vector=perturbed_vector)
        raw_score = origin_prob[true_y] - perturbed_prob[true_y]
        print('Prob before: ', origin_prob[true_y], '. Prob after: ', perturbed_prob[true_y],
              '. Prob shift: ', raw_score)
    return perturbed_text, perturbed_y, sub_rate, NE_rate, change_tuple_list



def adversarial_paraphrase_snli(opt, input_text_p, input_text_h, true_y, grad_guide, tokenizer, dataset, level, verbose=True):
    '''
    Compute a perturbation, greedily choosing the synonym if it causes the most
    significant change in the classification probability after replacement
    :return perturbed_text: generated adversarial examples
    :return perturbed_y: predicted class of perturbed_text
    :return sub_rate: word replacement rate showed in Table 3
    :return change_tuple_list: list of substitute words
    '''

    def halt_condition_fn(perturbed_text):
        '''
        Halt if model output is changed.
        '''
        perturbed_vector = None
        if level == 'word':
            perturbed_vector = text_to_vector(perturbed_text, tokenizer, dataset)
        elif level == 'char':
            max_len = config.char_max_len[dataset]
            perturbed_vector = doc_process(perturbed_text, get_embedding_dict(), dataset).reshape(1, max_len)
        adv_y = grad_guide.predict_classes(input_vector=perturbed_vector)
        if adv_y != true_y:
            return True
        else:
            return False

    def gen(perturbed_text_list):
        perturbed_text = ""
        recur = 0
        reduc = 0
        for i, word_str in enumerate(perturbed_text_list):
    
            if reduc==1 or i==0:
                space = ""
                reduc=0
            else:
                space = " "

            if len(word_str)==1 and word_str[0] in [".", ",", "-", ":", "!", "?", "(", ")", ";", "<", ">", "{","}", "[","]"]:
                space = ""
                if word_str[0] in [ "(", "<", "{", "["]:
                    reduc=1
            elif len(word_str)==1 and word_str[0] in ["\"",]:
                if recur==0:
                    space = " "
                    reduc=1
                elif recur==1:
                    space = ""
                recur=(recur+1)%2
            elif len(word_str)==1 and word_str[0] in ["'",]:
                space = ""
                reduc=1
            
            perturbed_text+=(space+word_str)

        return perturbed_text

    def heuristic_fn(text_p, text_h, candidate_h):
        '''
        Return the difference between the classification probability of the original
        word and the candidate substitute synonym, which is defined in Eq.(4) and Eq.(5).
        '''
        doc_h = nlp(text_h)

        origin_vector_h = None
        perturbed_vector_h = None
        if level == 'word':
            origin_vector_p = text_to_vector(text_p, tokenizer, dataset)
            origin_vector_h = text_to_vector(text_h, tokenizer, dataset)

            perturbed_text_list_h = _compile_perturbed_tokens(doc_h, [candidate_h])
            """
            perturbed_text = "" 
            for i, word_str in enumerate(perturbed_text_list):
                if i==0:
                    perturbed_text+=word_str
                else:
                    if word_str[0] in [".", ",", "-", "'", ":", "!", "?", "(", ")", ";", "<", ">"]:
                        perturbed_text+=word_str
                    else:
                        perturbed_text+=(" "+word_str)
            """
            perturbed_text_h = gen(perturbed_text_list_h)

            perturbed_doc_h = nlp(perturbed_text_h)
            perturbed_vector_h = text_to_vector(perturbed_doc_h.text, tokenizer, dataset)

        origin_prob = grad_guide.predict_prob(input_vector_p=origin_vector_p, input_vector_h=origin_vector_h)
        perturbed_prob = grad_guide.predict_prob(input_vector_p=origin_vector_p, input_vector_h=perturbed_vector_h)
        delta_p = origin_prob[true_y] - perturbed_prob[true_y]

        return delta_p

    doc_p = nlp(input_text_p)
    doc_h = nlp(input_text_h)

    # PWWS
    position_word_list_h, word_saliency_list_h = evaluate_word_saliency_snli(doc_p, doc_h, grad_guide, tokenizer, true_y, dataset, level)
    perturbed_text_p, perturbed_text_h, sub_rate, NE_rate, change_tuple_list = PWWS_snli(opt, doc_p, doc_h,
                                                                true_y,
                                                                dataset,
                                                                word_saliency_list=word_saliency_list_h,
                                                                heuristic_fn=heuristic_fn,
                                                                #halt_condition_fn=halt_condition_fn,
                                                                halt_condition_fn=None,
                                                                verbose=verbose)

    origin_vector = perturbed_vector = None
    if level == 'word':
        origin_vector_p = text_to_vector(input_text_p, tokenizer, dataset)
        perturbed_vector_p = text_to_vector(perturbed_text_p, tokenizer, dataset)
        origin_vector_h = text_to_vector(input_text_h, tokenizer, dataset)
        perturbed_vector_h = text_to_vector(perturbed_text_h, tokenizer, dataset)
    perturbed_y = grad_guide.predict_classes(input_vector_p=perturbed_vector_p, input_vector_h=perturbed_vector_h)
    if verbose:
        origin_prob = grad_guide.predict_prob(input_vector_p=origin_vector_p, input_vector_h=origin_vector_h)
        perturbed_prob = grad_guide.predict_prob(input_vector_p=perturbed_vector_p, input_vector_h=perturbed_vector_h)
        raw_score = origin_prob[true_y] - perturbed_prob[true_y]
        print('Prob before: ', origin_prob[true_y], '. Prob after: ', perturbed_prob[true_y],
              '. Prob shift: ', raw_score)
    return perturbed_text_p, perturbed_text_h, perturbed_y, sub_rate, NE_rate, change_tuple_list
