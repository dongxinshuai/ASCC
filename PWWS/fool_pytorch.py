# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import argparse
import os
import numpy as np
from .read_files import split_imdb_files, split_yahoo_files, split_agnews_files, split_snli_files
from .word_level_process import word_process, get_tokenizer
from .char_level_process import char_process
from .adversarial_tools import ForwardGradWrapper, ForwardGradWrapper_pytorch, adversarial_paraphrase, ForwardGradWrapper_pytorch_snli, adversarial_paraphrase_snli
import time
from .unbuffered import Unbuffered

try:
    import cPickle as pickle
except ImportError:
    import pickle

sys.stdout = Unbuffered(sys.stdout)

def write_origin_input_texts(origin_input_texts_path, test_texts, test_samples_cap=None):
    if test_samples_cap is None:
        test_samples_cap = len(test_texts)
    with open(origin_input_texts_path, 'a') as f:
        for i in range(test_samples_cap):
            f.write(test_texts[i] + '\n')

def genetic_attack(opt, device, model, attack_surface, dataset='imdb', genetic_test_num=100, test_bert=False):

    if test_bert:
        from modified_bert_tokenizer import ModifiedBertTokenizer
        tokenizer = ModifiedBertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    else:
        # get tokenizer
        tokenizer = get_tokenizer(opt)

    # Read data set
    x_test = y_test = None
    test_texts = None

    """
    if opt.synonyms_from_file:

        if dataset == 'imdb':
            train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = split_imdb_files(opt)
        elif dataset == 'agnews':
            train_texts, train_labels, test_texts, test_labels = split_agnews_files()
        elif dataset == 'yahoo':
            train_texts, train_labels, test_texts, test_labels = split_yahoo_files()

        filename= opt.synonyms_file_path
        f=open(filename,'rb')
        saved=pickle.load(f)
        f.close()
        #syn_data = saved["syn_data"]
        #opt.embeddings=saved['embeddings']
        #opt.vocab_size=saved['vocab_size']
        x_train=saved['x_train']
        x_test=saved['x_test']
        y_train=saved['y_train']
        y_test=saved['y_test']

    else:
        if dataset == 'imdb':
            train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = split_imdb_files(opt)
            x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)

        elif dataset == 'agnews':
            train_texts, train_labels, test_texts, test_labels = split_agnews_files()
            x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)

        elif dataset == 'yahoo':
            train_texts, train_labels, test_texts, test_labels = split_yahoo_files()
            x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)
    """

    if dataset == 'imdb':
        train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = split_imdb_files(opt)
        #x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)

    elif dataset == 'agnews':
        train_texts, train_labels, test_texts, test_labels = split_agnews_files()
        #x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)

    elif dataset == 'yahoo':
        train_texts, train_labels, test_texts, test_labels = split_yahoo_files()
        #x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)

    from .attacks import GeneticAdversary, AdversarialModel
    adversary = GeneticAdversary(attack_surface, num_iters=opt.genetic_iters, pop_size=opt.genetic_pop_size)

    from .config import config
    wrapped_model = AdversarialModel(model, tokenizer, config.word_max_len[dataset], test_bert=test_bert)

    adv_acc = adversary.run(wrapped_model, test_texts, test_labels, device, genetic_test_num, opt)
    print("genetic attack results:", adv_acc)
    return adv_acc

def genetic_attack_snli(opt, device, model, attack_surface, dataset='snli', genetic_test_num=100, split="test"):
    
    # get tokenizer
    tokenizer = get_tokenizer(opt)

    # Read data set
    x_test = y_test = None
    test_texts = None

    if dataset == 'snli':
        train_perms, train_hypos, train_labels, dev_perms, dev_hypos, dev_labels, test_perms, test_hypos, test_labels = split_snli_files(opt)
    else:
        raise NotImplementedError

    if split=="test":
        perms = test_perms
        hypos = test_hypos
    elif split=="dev":
        perms = dev_perms
        hypos = dev_hypos

    from .attacks import GeneticAdversary_Snli, AdversarialModel_Snli
    adversary = GeneticAdversary_Snli(attack_surface, num_iters=opt.genetic_iters, pop_size=opt.genetic_pop_size)

    from .config import config
    wrapped_model = AdversarialModel_Snli(model, tokenizer, config.word_max_len[dataset])

    adv_acc = adversary.run(wrapped_model, test_perms, test_hypos, test_labels, device, genetic_test_num, opt)
    print("genetic attack results:", adv_acc)

    return adv_acc


def fool_text_classifier_pytorch(opt, device,model, dataset='imdb', clean_samples_cap=50):
    print('clean_samples_cap:', clean_samples_cap)

    # get tokenizer
    tokenizer = get_tokenizer(opt)

    # Read data set
    x_test = y_test = None
    test_texts = None

    if opt.synonyms_from_file:

        if dataset == 'imdb':
            train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = split_imdb_files(opt)
        elif dataset == 'agnews':
            train_texts, train_labels, test_texts, test_labels = split_agnews_files()
        elif dataset == 'yahoo':
            train_texts, train_labels, test_texts, test_labels = split_yahoo_files()

        filename= opt.imdb_synonyms_file_path
        f=open(filename,'rb')
        saved=pickle.load(f)
        f.close()
        #syn_data = saved["syn_data"]
        #opt.embeddings=saved['embeddings']
        #opt.vocab_size=saved['vocab_size']
        x_train=saved['x_train']
        x_test=saved['x_test']
        y_train=saved['y_train']
        y_test=saved['y_test']

    else:
        if dataset == 'imdb':
            train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = split_imdb_files(opt)
            x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)

        elif dataset == 'agnews':
            train_texts, train_labels, test_texts, test_labels = split_agnews_files()
            x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)

        elif dataset == 'yahoo':
            train_texts, train_labels, test_texts, test_labels = split_yahoo_files()
            x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)

    grad_guide = ForwardGradWrapper_pytorch(model, device)
    classes_prediction = grad_guide.predict_classes(x_test[: clean_samples_cap])

    print('Crafting adversarial examples...')
    successful_perturbations = 0
    failed_perturbations = 0
    all_test_num =0

    sub_rate_list = []
    NE_rate_list = []

    start_cpu = time.clock()
    fa_path = r'./fool_result/{}'.format(dataset)
    if not os.path.exists(fa_path):
        os.makedirs(fa_path)
    adv_text_path = r'./fool_result/{}/adv_{}.txt'.format(dataset, str(clean_samples_cap))
    change_tuple_path = r'./fool_result/{}/change_tuple_{}.txt'.format(dataset, str(clean_samples_cap))
    #file_1 = open(adv_text_path, "a")
    #file_2 = open(change_tuple_path, "a")

    for index, text in enumerate(test_texts[opt.h_test_start: opt.h_test_start+clean_samples_cap]):
        sub_rate = 0
        NE_rate = 0
        all_test_num+=1
        print('_____{}______.'.format(index))
        if np.argmax(y_test[index]) == classes_prediction[index]:
        #if True:
            print('do')
            # If the ground_true label is the same as the predicted label
            adv_doc, adv_y, sub_rate, NE_rate, change_tuple_list = adversarial_paraphrase(opt,
                                                                                          input_text=text,
                                                                                          true_y=np.argmax(y_test[index]),
                                                                                          grad_guide=grad_guide,
                                                                                          tokenizer=tokenizer,
                                                                                          dataset=dataset,
                                                                                          level='word')
            if adv_y != np.argmax(y_test[index]):
                successful_perturbations += 1
                print('{}. Successful example crafted.'.format(index))
            else:
                failed_perturbations += 1
                print('{}. Failure.'.format(index))

            print("r acc", 1.0*failed_perturbations/all_test_num)

            text = adv_doc
            sub_rate_list.append(sub_rate)
            NE_rate_list.append(NE_rate)
            #file_2.write(str(index) + str(change_tuple_list) + '\n')
        #file_1.write(text + " sub_rate: " + str(sub_rate) + "; NE_rate: " + str(NE_rate) + "\n")
    end_cpu = time.clock()
    print('CPU second:', end_cpu - start_cpu)

    #mean_sub_rate = sum(sub_rate_list) / len(sub_rate_list)
    #mean_NE_rate = sum(NE_rate_list) / len(NE_rate_list)
    print('substitution:', sum(sub_rate_list))
    print('sum substitution:', len(sub_rate_list))
    print('NE rate:', sum(NE_rate_list))
    print('sum NE:', len(NE_rate_list))
    print("succ attack %d"%(successful_perturbations))
    print("fail attack %d"%(failed_perturbations))
    #file_1.close()
    #file_2.close()




def fool_text_classifier_pytorch_snli(opt, device,model, dataset='imdb', clean_samples_cap=50):
    print('clean_samples_cap:', clean_samples_cap)

    # get tokenizer
    tokenizer = get_tokenizer(opt)

    # Read data set
    x_test = y_test = None
    test_texts = None

    if dataset == 'snli':
        train_perms, train_hypos, train_labels, dev_perms, dev_hypos, dev_labels, test_perms, test_hypos, test_labels = split_snli_files(opt)
    else:
        raise NotImplementedError

    assert(opt.synonyms_from_file)
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

    grad_guide = ForwardGradWrapper_pytorch_snli(model, device)
    classes_prediction = grad_guide.predict_classes(x_p_test[: clean_samples_cap], x_h_test[: clean_samples_cap])

    print('Crafting adversarial examples...')
    successful_perturbations = 0
    failed_perturbations = 0
    all_test_num =0

    sub_rate_list = []
    NE_rate_list = []

    start_cpu = time.clock()

    for index in range(opt.h_test_start, opt.h_test_start+clean_samples_cap, 1):

        text_p=test_perms[index]
        text_h=test_hypos[index]

        sub_rate = 0
        NE_rate = 0
        all_test_num+=1
        print('_____{}______.'.format(index))
        if np.argmax(y_test[index]) == classes_prediction[index]:
            print('do')
            # If the ground_true label is the same as the predicted label
            adv_doc_p, adv_doc_h, adv_y, sub_rate, NE_rate, change_tuple_list = adversarial_paraphrase_snli(opt, input_text_p=text_p, input_text_h=text_h,
                                                                                          true_y=np.argmax(y_test[index]),
                                                                                          grad_guide=grad_guide,
                                                                                          tokenizer=tokenizer,
                                                                                          dataset=dataset,
                                                                                          level='word')
            if adv_y != np.argmax(y_test[index]):
                successful_perturbations += 1
                print('{}. Successful example crafted.'.format(index))
            else:
                failed_perturbations += 1
                print('{}. Failure.'.format(index))

        print("r acc", 1.0*failed_perturbations/all_test_num)

        sub_rate_list.append(sub_rate)
        NE_rate_list.append(NE_rate)

    end_cpu = time.clock()
    print('CPU second:', end_cpu - start_cpu)
    print("PWWS acc:", 1.0*failed_perturbations/all_test_num)


    #print('substitution:', sum(sub_rate_list))
    #print('sum substitution:', len(sub_rate_list))
    #print('NE rate:', sum(NE_rate_list))
    #print('sum NE:', len(NE_rate_list))
    #print("succ attack %d"%(successful_perturbations))
    #print("fail attack %d"%(failed_perturbations))

