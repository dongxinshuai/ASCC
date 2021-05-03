# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from six.moves import cPickle
import time,os,random
import itertools

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import NLLLoss,MultiLabelSoftMarginLoss,MultiLabelMarginLoss,BCELoss

from dataHelper import load_synonyms_in_vocab, imdb_make_synthesized_iter, snli_make_synthesized_iter
from PWWS.fool_pytorch import fool_text_classifier_pytorch, genetic_attack

import opts
import models
import utils

from solver.lr_scheduler import WarmupMultiStepLR

try:
    import cPickle as pickle
except ImportError:
    import pickle
      

def set_params(net, resume_model_path, data_parallel=False):
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(resume_model_path), 'Error: ' + resume_model_path + 'checkpoint not found!'
    checkpoint = torch.load(resume_model_path)
    state_dict = checkpoint['net']
    from collections import OrderedDict
    sdict = OrderedDict()
    for key in state_dict.keys():
        new_key = key.split('module.')[-1]
        if data_parallel:
            new_key = 'module.'+new_key
        sdict[new_key]=state_dict[key]
    net.load_state_dict(sdict)
    return net


def train(opt,train_iter, dev_iter, test_iter, syn_data, verbose=True):
    global_start= time.time()
    #logger = utils.getLogger()
    model=models.setup(opt)

    if opt.resume != None:
        model = set_params(model, opt.resume)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        model.cuda()
        #model=torch.nn.DataParallel(model)

    # set optimizer
    if opt.embd_freeze == True:
        model.embedding.weight.requires_grad = False
    else:
        model.embedding.weight.requires_grad = True
    params = [param for param in model.parameters() if param.requires_grad] #filter(lambda p: p.requires_grad, model.parameters())
    optimizer = utils.getOptimizer(params,name=opt.optimizer, lr=opt.learning_rate,weight_decay=opt.weight_decay,scheduler= utils.get_lr_scheduler(opt.lr_scheduler))
    scheduler = WarmupMultiStepLR(optimizer, (40, 80), 0.1, 1.0/10.0, 2, 'linear')

    from label_smooth import LabelSmoothSoftmaxCE
    if opt.label_smooth!=0:
        assert(opt.label_smooth<=1 and opt.label_smooth>0)
        loss_fun = LabelSmoothSoftmaxCE(lb_pos=1-opt.label_smooth, lb_neg=opt.label_smooth)
    else:
        loss_fun = F.cross_entropy

    filename = None
    acc_adv_list=[]
    start= time.time()
    kl_control = 0

    # initialize synonyms with the same embd
    from PWWS.word_level_process import word_process, get_tokenizer
    tokenizer = get_tokenizer(opt)

    if opt.embedding_prep == "same":
        father_dict= {}
        for index in range(1+len(tokenizer.index_word)):
            father_dict[index] = index

        def get_father(x):
            if father_dict[x] == x:
                return x
            else:
                fa = get_father(father_dict[x])
                father_dict[x] = fa
                return fa

        for index in range(len(syn_data)-1, 0, -1):
            syn_list = syn_data[index]
            for pos in syn_list:
                fa_pos = get_father(pos)
                fa_anch = get_father(index)
                if fa_pos == fa_anch:
                    father_dict[index] = index
                    father_dict[fa_anch] = index
                else:
                    father_dict[index] = index
                    father_dict[fa_anch] = index
                    father_dict[fa_pos] = index
                
        print("Same embedding for synonyms as embd prep.")
        set_different_embd=set()
        for key in father_dict:
            fa = get_father(key)
            set_different_embd.add(fa)
            with torch.no_grad():
                model.embedding.weight[key, :] = model.embedding.weight[fa, :]
        print(len(set_different_embd))

    elif opt.embedding_prep == "ge":
        print("Graph embedding as embd prep.")
        ge_file_path = opt.ge_file_path
        f=open(ge_file_path,'rb')
        saved=pickle.load(f)
        ge_embeddings_dict = saved['walk_embeddings']
        #model = saved['model']
        f.close()
        with torch.no_grad():
            for key in ge_embeddings_dict:
                model.embedding.weight[int(key), :] = torch.FloatTensor(ge_embeddings_dict[key])
    else:
        print("No embd prep.")

    from from_certified.attack_surface import WordSubstitutionAttackSurface, LMConstrainedAttackSurface
    if opt.lm_constraint:
        attack_surface = LMConstrainedAttackSurface.from_files(opt.certified_neighbors_file_path, opt.imdb_lm_file_path)
    else:
        attack_surface = WordSubstitutionAttackSurface.from_files(opt.certified_neighbors_file_path, opt.imdb_lm_file_path)

    best_adv_acc = 0
    for epoch in range(21):

        if opt.smooth_ce:
            if epoch < 10:
                weight_adv = epoch*1.0/10
                weight_clean = 1-weight_adv
            else:
                weight_adv=1
                weight_clean=0
        else:
            weight_adv = opt.weight_adv
            weight_clean = opt.weight_clean

        if epoch>=opt.kl_start_epoch:
            kl_control = 1

        sum_loss = sum_loss_adv = sum_loss_kl = sum_loss_clean = 0
        total = 0

        for iters,batch in enumerate(train_iter):

            text = batch[0].to(device)
            label = batch[1].to(device)
            anch = batch[2].to(device)
            pos = batch[3].to(device)
            neg = batch[4].to(device)
            anch_valid= batch[5].to(device).unsqueeze(2)
            text_like_syn= batch[6].to(device)
            text_like_syn_valid= batch[7].to(device)

            bs, sent_len = text.shape

            model.train()
            
            # zero grad
            optimizer.zero_grad()

            if opt.pert_set=="ad_text":
                attack_type_dict = {
                    'num_steps': opt.train_attack_iters,
                    'loss_func': 'ce' if opt.if_ce_adp else 'kl',
                    'w_optm_lr': opt.w_optm_lr,
                    'sparse_weight': opt.attack_sparse_weight,
                    'out_type': "text"
                }
                embd = model(mode="text_to_embd", input=text) #in bs, len sent, vocab
                n,l,s = text_like_syn.shape
                text_like_syn_embd = model(mode="text_to_embd", input=text_like_syn.reshape(n,l*s)).reshape(n,l,s,-1)
                text_adv = model(mode="get_adv_by_convex_syn", input=embd, label=label, text_like_syn_embd=text_like_syn_embd, text_like_syn_valid=text_like_syn_valid, text_like_syn=text_like_syn, attack_type_dict=attack_type_dict)
            
            elif opt.pert_set=="ad_text_syn_p":
                attack_type_dict = {
                    'num_steps': opt.train_attack_iters,
                    'loss_func': 'ce' if opt.if_ce_adp else 'kl',
                    'w_optm_lr': opt.w_optm_lr,
                    'sparse_weight': opt.train_attack_sparse_weight,
                    'out_type': "comb_p"
                }
                embd = model(mode="text_to_embd", input=text) #in bs, len sent, vocab
                n,l,s = text_like_syn.shape
                text_like_syn_embd = model(mode="text_to_embd", input=text_like_syn.reshape(n,l*s)).reshape(n,l,s,-1)
                adv_comb_p = model(mode="get_adv_by_convex_syn", input=embd, label=label, text_like_syn_embd=text_like_syn_embd, text_like_syn_valid=text_like_syn_valid, attack_type_dict=attack_type_dict)

            elif opt.pert_set=="ad_text_hotflip":
                attack_type_dict = {
                    'num_steps': opt.train_attack_iters,
                    'loss_func': 'ce' if opt.if_ce_adp else 'kl',
                }
                text_adv = model(mode="get_adv_hotflip", input=text, label=label, text_like_syn_valid=text_like_syn_valid, text_like_syn=text_like_syn, attack_type_dict=attack_type_dict)
            
            elif opt.pert_set=="l2_ball":
                set_radius = opt.train_attack_eps
                attack_type_dict = {
                    'num_steps': opt.train_attack_iters,
                    'step_size': opt.train_attack_step_size * set_radius,
                    'random_start': opt.random_start,
                    'epsilon':  set_radius,
                    #'loss_func': 'ce',
                    'loss_func': 'ce' if opt.if_ce_adp else 'kl',
                    'direction': 'away',
                    'ball_range': opt.l2_ball_range,
                }
                embd = model(mode="text_to_embd", input=text) #in bs, len sent, vocab
                embd_adv = model(mode="get_embd_adv", input=embd, label=label, attack_type_dict=attack_type_dict)

            optimizer.zero_grad()
            # clean loss
            predicted = model(mode="text_to_logit", input=text)
            loss_clean= loss_fun(predicted,label)
            # adv loss
            if opt.pert_set=="ad_text" or opt.pert_set=="ad_text_hotflip":
                predicted_adv = model(mode="text_to_logit", input=text_adv)
            elif opt.pert_set=="ad_text_syn_p":
                predicted_adv = model(mode="text_syn_p_to_logit", input=text_like_syn, comb_p=adv_comb_p)
            elif opt.pert_set=="l2_ball":
                predicted_adv = model(mode="embd_to_logit", input=embd_adv)

            loss_adv = loss_fun(predicted_adv,label)
            # kl loss
            criterion_kl = nn.KLDivLoss(reduction="sum")
            loss_kl = (1.0 / bs) * criterion_kl(F.log_softmax(predicted_adv, dim=1),
                                                        F.softmax(predicted, dim=1))

            # optimize
            loss =  opt.weight_kl * kl_control * loss_kl + weight_adv * loss_adv + weight_clean * loss_clean
            loss.backward() 
            optimizer.step()
            sum_loss += loss.item()
            sum_loss_adv += loss_adv.item()
            sum_loss_clean += loss_clean.item()
            sum_loss_kl += loss_kl.item()
            predicted, idx = torch.max(predicted, 1) 
            precision=(idx==label).float().mean().item()
            predicted_adv, idx = torch.max(predicted_adv, 1)
            precision_adv=(idx==label).float().mean().item()
            total += 1

            out_log = "%d epoch %d iters: loss: %.3f, loss_kl: %.3f, loss_adv: %.3f, loss_clean: %.3f | acc: %.3f acc_adv: %.3f | in %.3f seconds" % (epoch, iters, sum_loss/total, sum_loss_kl/total, sum_loss_adv/total, sum_loss_clean/total, precision, precision_adv, time.time()-start)
            start= time.time()
            print(out_log)
                

        scheduler.step()

        if epoch%1==0:
            acc=utils.imdb_evaluation(opt, device, model, dev_iter)
            out_log="%d epoch with dev acc %.4f" % (epoch,acc)
            print(out_log)
            adv_acc=utils.imdb_evaluation_ascc_attack(opt, device, model, dev_iter, tokenizer)
            out_log="%d epoch with dev adv acc against ascc attack %.4f" % (epoch,adv_acc)
            print(out_log)

            #hotflip_adv_acc=utils.evaluation_hotflip_adv(opt, device, model, dev_iter, tokenizer)
            #out_log="%d epoch with dev hotflip adv acc %.4f" % (epoch,hotflip_adv_acc)
            #logger.info(out_log)
            #print(out_log)

            if adv_acc>=best_adv_acc:
                best_adv_acc = adv_acc
                best_save_dir=os.path.join(opt.out_path, "{}_best.pth".format(opt.model))
                state = {
                    'net': model.state_dict(),
                    'epoch': epoch,
                }
                torch.save(state, best_save_dir)

    # restore best according to dev set
    model = set_params(model, best_save_dir)
    acc=utils.imdb_evaluation(opt, device, model, test_iter)
    print("test acc %.4f" % (acc))
    adv_acc=utils.imdb_evaluation_ascc_attack(opt, device, model, test_iter, tokenizer)
    print("test adv acc against ascc attack %.4f" % (adv_acc))
    genetic_attack(opt, device, model, attack_surface, dataset=opt.dataset, genetic_test_num=opt.genetic_test_num)
    fool_text_classifier_pytorch(opt, device, model, dataset=opt.dataset, clean_samples_cap=opt.pwws_test_num)
    
def main():
    opt = opts.parse_opt()
    print(opt)

    assert(opt.dataset=="imdb")
    train_iter, dev_iter, test_iter, syn_data = imdb_make_synthesized_iter(opt)

    train(opt, train_iter, dev_iter, test_iter, syn_data)
    
if __name__=="__main__": 
    main()