# -*- coding: utf-8 -*-
import os
import numpy as np

from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .CNNKim import *

try:
    import cPickle as pickle
except ImportError:
    import pickle
import sys
"""
def get_embd_adv(self, embd, y, attack_type_dict):
    # record context
    self_training_context = self.training
    # set context
    self.eval()
    
    func_bpda_get_embd_adv = bpda_get_embd_adv()

    embd_adv = func_bpda_get_embd_adv(embd, y, attack_type_dict, self.embd_to_logit, self.l2_project, self.l2_clip)

    # resume context
    if self_training_context == True:
        self.train()
    else:
        self.eval()

    return embd_adv
class bpda_get_embd_adv(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, embd, y, attack_type_dict, func_embd_to_logit, func_l2_project, func_l2_clip):

        device = embd.device
        # get param of attacks

        num_steps=attack_type_dict['num_steps']
        step_size=attack_type_dict['step_size']
        random_start=attack_type_dict['random_start']
        epsilon=attack_type_dict['epsilon']
        loss_func=attack_type_dict['loss_func']
        direction=attack_type_dict['direction']

        batch_size=len(embd)

        embd_ori = embd
        
        # random start
        if random_start:
            embd_adv = embd_ori.detach() + 0.001 * torch.randn(embd_ori.shape).to(device).detach()
        else:
            embd_adv = embd_ori.detach()


        for _ in range(num_steps):
            embd_adv.requires_grad_()
            grad = 0
            with torch.enable_grad():
                if loss_func=='ce':
                    logit_adv = func_embd_to_logit(embd_adv)
                    if direction == "towards":
                        loss = -F.cross_entropy(logit_adv, y, reduction='sum')
                    elif direction == "away":
                        loss = F.cross_entropy(logit_adv, y, reduction='sum')
                grad = torch.autograd.grad(loss, [embd_adv])[0]

            grad=func_l2_project(grad)

            embd_adv = embd_adv.detach() + step_size * grad.detach()
            
            perturbation = func_l2_clip(embd_adv-embd_ori, epsilon)
            embd_adv = embd_ori.detach() + perturbation.detach()
            
        return embd_adv.detach()


    @staticmethod
    def backward(ctx, grad_output):
        # BPDA, approximate gradients
        return grad_output
"""


class AdvBaseModel(BaseModel):
    def __init__(self, opt, is_bert=False):
        super(AdvBaseModel, self).__init__(opt,is_bert=is_bert)
        self.opt=opt
        if is_bert:
            return

        """
        # inverse embedding
        print("making inverse embedding")
        
        #inverse_embedding_weight = self.embedding.weight.detach().cpu()
        #inverse_embedding_weight = inverse_embedding_weight.numpy()
        #inverse_embedding_weight = np.matrix(inverse_embedding_weight)
        #inverse_embedding_weight = np.linalg.pinv(inverse_embedding_weight)
        #inverse_embedding_weight = torch.FloatTensor(inverse_embedding_weight)
        
        #self.inverse_embedding = nn.Embedding(self.embedding_dim, self.vocab_size + 2, ) 
        self.inverse_embedding = nn.Linear( self.vocab_size + 2,self.embedding_dim, bias=False)
        
        #self.inverse_embedding.weight=nn.Parameter(inverse_embedding_weight)            
        #self.inverse_embedding.weight.requires_grad = False
        
        
        # embd to embdnew
        self.new_embedding_dim = self.embedding_dim
        self.linear_transform_embd = nn.Linear(self.embedding_dim, self.new_embedding_dim)

        # embdnew to embd
        self.inverse_linear_transform_embd = nn.Linear(self.new_embedding_dim, self.embedding_dim)
        #self.update_linear_transform_embd()
        """
        self.embedding_dim = opt.embedding_dim
        self.vocab_size = opt.vocab_size
        self.embedding = nn.Embedding(opt.vocab_size+2,opt.embedding_dim)
        if opt.use_pretrained_embeddings:
            self.embedding.weight=nn.Parameter(opt.embeddings) 
        self.embedding.weight.requires_grad = True

        if self.opt.normalize_embedding:
            with torch.no_grad():
                self.embedding.weight = nn.Parameter(self.normalize_embedding(self.embedding.weight, self.opt.vocab_freq))

        self.label_size=opt.label_size

        self.embedding_out_dim = self.embedding_dim
        if self.opt.embd_transform:
            self.embedding_out_dim = opt.embedding_out_dim
            self.linear_transform_embd_1 = nn.Linear(self.embedding_dim, self.embedding_out_dim)
            #self.linear_transform_embd_2 = nn.Linear(self.latent_dim, self.latent_dim)
            #self.linear_transform_embd_3 = nn.Linear(self.latent_dim, self.embedding_dim)

        # l2 radius 
        self.word_synonym_radius = nn.Embedding(self.vocab_size + 2, 1) #, padding_idx=self.vocab_size + 1
        self.word_synonym_radius.weight.requires_grad = False
        self.word_synonym_radius.weight=nn.Parameter(torch.zeros_like(self.word_synonym_radius.weight))

        self.eval_adv_mode = True

    def normalize_embedding(self, weight, vocab_freq):
        # input torch tensor
        p = vocab_freq / vocab_freq.sum()
        mean = torch.sum(p.unsqueeze(1)*weight, dim=0).unsqueeze(0)
        var = torch.sum(p.unsqueeze(1)*torch.pow(weight-mean,2), dim=0).unsqueeze(0)
        stddev = torch.sqrt(var+1e-6)
        return p.unsqueeze(1)*(weight-mean)/stddev # if p==0, use zero vector

    def update_linear_transform_embd(self):
        weight=self.inverse_linear_transform_embd.weight.detach().cpu() # W x +b 
        bias=self.inverse_linear_transform_embd.bias.detach().cpu()

        weight = weight.numpy()
        weight = np.matrix(weight)
        weight = np.linalg.pinv(weight)
        weight = torch.FloatTensor(weight)

        self.linear_transform_embd.weight=nn.Parameter(weight)  
        bias = -bias.reshape(1,-1).mm(weight.transpose(0,1)) 
        bias = bias.squeeze()
        self.linear_transform_embd.bias=nn.Parameter(bias) 

    def update_inverse_linear_transform_embd(self):
        weight=self.linear_transform_embd.weight.detach().cpu() # W x +b 
        bias=self.linear_transform_embd.bias.detach().cpu()

        weight = weight.numpy()
        weight = np.matrix(weight)
        weight = np.linalg.pinv(weight)
        weight = torch.FloatTensor(weight)

        self.inverse_linear_transform_embd.weight=nn.Parameter(weight)  
        bias = -bias.reshape(1,-1).mm(weight.transpose(0,1)) 
        bias = bias.squeeze()
        self.inverse_linear_transform_embd.bias=nn.Parameter(bias) 


    def loss_radius(self, roots, synonyms):
        embd_roots = self.text_to_embd(roots.reshape(-1,1))
        embd_synonyms = self.text_to_embd(synonyms)
        n, len_syn_set, embd_dim = embd_synonyms.shape
        delta = (embd_synonyms-embd_roots).reshape(-1, embd_dim)

        radius = torch.nn.functional.pairwise_distance(delta, torch.zeros_like(delta), p=2.0) # n*len_syn_set,
        max_radius, _ = radius.reshape(n, len_syn_set).max(-1)
        
        for i in range(n):
            self.word_synonym_radius.weight[roots[i],0]=max_radius[i]
        
        print(max_radius.mean())

        #saved_radius = self.word_synonym_radius(roots).squeeze() # n
        #return torch.nn.functional.mse_loss(max_radius, saved_radius, reduction='mean')

    def l2_project(self, grad):

        N, L, E = grad.shape

        torch_small_constant = 1e-12*torch.ones(N,L,1).to(grad.dtype).to(grad.device)

        grad_norm=grad*grad
        grad_norm=torch.sum(grad_norm, dim=-1, keepdim=True)
        grad_norm = torch.sqrt(grad_norm)
        grad_norm = torch.max(torch_small_constant, grad_norm)
        grad = grad / grad_norm

        return grad

    def l2_project_sent(self, grad):
    
        N, L, E = grad.shape

        torch_small_constant = 1e-12*torch.ones(N,1,1).to(grad.dtype).to(grad.device)

        grad_norm=grad*grad
        grad_norm=torch.sum(grad_norm, dim=(1,2), keepdim=True)
        grad_norm = torch.sqrt(grad_norm)
        grad_norm = torch.max(torch_small_constant, grad_norm)
        grad = grad / grad_norm

        return grad

    def l2_clip(self, pert, eps):
    
        N, L, E = pert.shape
        torch_small_constant = 1e-12*torch.ones(N,L,1).to(pert.dtype).to(pert.device)
        
        torch_ones = torch.ones_like(pert)

        pert_norm=pert*pert
        pert_norm=torch.sum(pert_norm, dim=-1, keepdim=True)
        pert_norm = torch.sqrt(pert_norm)
        pert_norm = torch.max(torch_small_constant, pert_norm)
        ratio = eps / pert_norm
        ratio = torch.min(torch_ones, ratio)

        return pert*ratio

    def l2_clip_sent(self, pert, eps):
        
        N, L, E = pert.shape
        torch_small_constant = 1e-12*torch.ones(N,1,1).to(pert.dtype).to(pert.device)
        
        torch_ones = torch.ones_like(pert)

        pert_norm=pert*pert
        pert_norm=torch.sum(pert_norm, dim=(1,2), keepdim=True)
        pert_norm = torch.sqrt(pert_norm)
        pert_norm = torch.max(torch_small_constant, pert_norm)
        ratio = eps / pert_norm
        ratio = torch.min(torch_ones, ratio)

        return pert*ratio

    def get_adv_by_convex_syn(self, embd, y, syn, syn_valid, text_like_syn, attack_type_dict, text_for_vis, record_for_vis):
        
        # record context
        self_training_context = self.training
        # set context
        if self.eval_adv_mode:
            self.eval()
        else:
            self.train()

        device = embd.device
        # get param of attacks

        num_steps=attack_type_dict['num_steps']
        loss_func=attack_type_dict['loss_func']
        w_optm_lr=attack_type_dict['w_optm_lr']
        sparse_weight = attack_type_dict['sparse_weight']
        out_type = attack_type_dict['out_type']

        batch_size, text_len, embd_dim = embd.shape
        batch_size, text_len, syn_num, embd_dim = syn.shape

        w = torch.empty(batch_size, text_len, syn_num, 1).to(device).to(embd.dtype)
        #ww = torch.zeros(batch_size, text_len, syn_num, 1).to(device).to(embd.dtype)
        #ww = ww+500*(syn_valid.reshape(batch_size, text_len, syn_num, 1)-1)
        nn.init.kaiming_normal_(w)
        w.requires_grad_()
        
        import utils
        params = [w] 
        optimizer = utils.getOptimizer(params,name='adam', lr=w_optm_lr,weight_decay=2e-5)

        def get_comb_p(w, syn_valid):
            ww=w*syn_valid.reshape(batch_size, text_len, syn_num, 1) + 500*(syn_valid.reshape(batch_size, text_len, syn_num, 1)-1)
            return F.softmax(ww, -2)

        def get_comb_ww(w, syn_valid):
            ww=w*syn_valid.reshape(batch_size, text_len, syn_num, 1) + 500*(syn_valid.reshape(batch_size, text_len, syn_num, 1)-1)
            return ww

        def get_comb(p, syn):
            return (p* syn.detach()).sum(-2)


        embd_ori=embd.detach()
        with torch.no_grad():
            logit_ori = self.embd_to_logit(embd_ori)

        for _ in range(num_steps):
            optimizer.zero_grad()
            with torch.enable_grad():
                ww = get_comb_ww(w, syn_valid)
                #comb_p = get_comb_p(w, syn_valid)
                embd_adv = get_comb(F.softmax(ww, -2), syn)
                if loss_func=='ce':
                    logit_adv = self.embd_to_logit(embd_adv)
                    loss = -F.cross_entropy(logit_adv, y, reduction='sum')
                elif loss_func=='kl':
                    logit_adv = self.embd_to_logit(embd_adv)
                    criterion_kl = nn.KLDivLoss(reduction="sum")
                    loss = -criterion_kl(F.log_softmax(logit_adv, dim=1),
                                        F.softmax(logit_ori.detach(), dim=1))

                #print("ad loss:", loss.data.item())
                                    
                if sparse_weight !=0:
                    #loss_sparse = (comb_p*comb_p).mean()
                    loss_sparse = (-F.softmax(ww, -2)*F.log_softmax(ww, -2)).sum(-2).mean()
                    #loss -= sparse_weight*loss_sparse
                    
                    loss = loss + sparse_weight*loss_sparse
                    #print(loss_sparse.data.item())

            #loss*=1000
            loss.backward()
            optimizer.step()

        #print((ww-w).max())

        comb_p = get_comb_p(w, syn_valid)

        if self.opt.vis_w_key_token is not None:
            assert(text_for_vis is not None and record_for_vis is not None)
            vis_n, vis_l = text_for_vis.shape
            for i in range(vis_n):
                for j in range(vis_l):
                    if text_for_vis[i,j] == self.opt.vis_w_key_token:
                        record_for_vis["comb_p_list"].append(comb_p[i,j].cpu().detach().numpy())
                        record_for_vis["embd_syn_list"].append(syn[i,j].cpu().detach().numpy())
                        record_for_vis["syn_valid_list"].append(syn_valid[i,j].cpu().detach().numpy())
                        record_for_vis["text_syn_list"].append(text_like_syn[i,j].cpu().detach().numpy())
                        
                        print("record for vis", len(record_for_vis["comb_p_list"]))
                    if len(record_for_vis["comb_p_list"])>=300:
                        dir_name = self.opt.resume.split(self.opt.model)[0]
                        file_name = self.opt.dataset+"_vis_w_"+str(self.opt.attack_sparse_weight)+"_"+str(self.opt.vis_w_key_token)+".pkl"
                        file_name = os.path.join(dir_name, file_name)
                        f=open(file_name,'wb')
                        pickle.dump(record_for_vis, f)
                        f.close()
                        sys.exit()
                        

        """
        out = get_comb(comb_p, syn)
        delta = (out-embd_ori).reshape(batch_size*text_len,embd_dim)
        delta = F.pairwise_distance(delta, torch.zeros_like(delta), p=2.0)
        valid = (delta>0.01).to(device).to(delta.dtype)
        delta = (valid*delta).sum()/valid.sum()
        print("mean l2 dis between embd and embd_adv:", delta.data.item())
        #print("mean max comb_p:", (comb_p.max(-2)[0]).mean().data.item())
        """

        if out_type == "text":
            # need to be fix, has potential bugs. the trigger dependes on data.
            assert(text_like_syn is not None) # n l synlen
            comb_p = comb_p.reshape(batch_size* text_len, syn_num)
            ind = comb_p.max(-1)[1] # shape batch_size* text_len
            out = (text_like_syn.reshape(batch_size* text_len, syn_num)[np.arange(batch_size*text_len), ind]).reshape(batch_size, text_len)
        elif out_type == "comb_p":
            out = comb_p

        # resume context
        if self_training_context == True:
            self.train()
        else:
            self.eval()

        return out.detach()

    def get_onehot_from_input(self, x):
        embd_voc_dim,  embd_dim= self.embedding.weight.shape

        bs, text_len = x.shape

        out = torch.zeros(bs*text_len, embd_voc_dim).to(torch.float32).to(x.device).scatter_(1, x.reshape(bs*text_len, 1), 1)

        return out.reshape(bs, text_len, embd_voc_dim)

    def get_onehot_mask_from_syn(self, syn_valid, syn):
        embd_voc_dim,  embd_dim= self.embedding.weight.shape

        bs, text_len, syn_max_num = syn.shape

        out = torch.zeros(bs*text_len, embd_voc_dim).to(torch.float32).to(syn.device).scatter_(1, syn.reshape(bs*text_len,syn_max_num), 1)
        out[:,0] = 0

        return out.reshape(bs, text_len, embd_voc_dim)

    def get_embd_from_onehot(self, onehot_input):
        w = self.embedding.weight
        bs, text_len, voc_d = onehot_input.shape
        embd=torch.mm(onehot_input.reshape(bs*text_len, voc_d), w)
        embd= embd.reshape(bs,text_len, -1)

        if self.opt.embd_transform:
            embd = self.linear_transform_embd_1(embd)
        return embd

    def get_adv_hotflip(self, x, y, syn_valid, text_like_syn, attack_type_dict):
        
        # record context
        self_training_context = self.training
        # set context
        if self.eval_adv_mode:
            self.eval()
        else:
            self.train()

        device = x.device
        # get param of attacks

        num_steps=attack_type_dict['num_steps']
        loss_func=attack_type_dict['loss_func']

        batch_size, text_len = x.shape

        #onehot_input = self.get_onehot_from_input(x) #bs, len, len_of_vocab
        onehot_mask = self.get_onehot_mask_from_syn(syn_valid, text_like_syn)
        #9.25 04:09
        #embd = self.get_embd_from_onehot(onehot_input)
        embd = self.text_to_embd(x)
        logit = self.embd_to_logit(embd)

        x_adv = torch.zeros_like(x)
        x_adv = x.detach()

        """
        onehot_input_adv = self.get_onehot_from_input(x) #bs, len, len_of_vocab
        onehot_input_adv.requires_grad_()
        with torch.enable_grad():
            embd_temp = self.get_embd_from_onehot(onehot_input_adv)
            logit_temp = self.embd_to_logit(embd_temp)
            if loss_func=='ce':
                loss = F.cross_entropy(logit_temp, y, reduction='sum')
            elif loss_func=='kl':
                criterion_kl = nn.KLDivLoss(reduction="sum")
                loss = criterion_kl(F.log_softmax(logit_temp, dim=1), F.softmax(logit, dim=1))

            grad = torch.autograd.grad(loss, [onehot_input_adv])[0]
        """

        for i in range(num_steps):
            embd_adv = self.text_to_embd(x_adv)
            embd_adv.requires_grad_()
            with torch.enable_grad():
                logit_adv = self.embd_to_logit(embd_adv)
                if loss_func=='ce':
                    loss = F.cross_entropy(logit_adv, y, reduction='sum')
                elif loss_func=='kl':
                    criterion_kl = nn.KLDivLoss(reduction="sum")
                    loss = criterion_kl(F.log_softmax(logit_adv, dim=1), F.softmax(logit, dim=1))

                grad_embd = torch.autograd.grad(loss, [embd_adv])[0]

            with torch.no_grad():
                if self.opt.embd_transform:
                    grad_onehot = torch.mm( grad_embd.reshape(batch_size*text_len,-1), torch.mm(self.embedding.weight, self.linear_transform_embd_1.weight.permute(1,0)).permute(1,0) )
                else:
                    grad_onehot = torch.mm( grad_embd.reshape(batch_size*text_len,-1), self.embedding.weight.permute(1,0) )
                grad_onehot =  grad_onehot.reshape(batch_size, text_len, -1) * (onehot_mask)
                _, argmax = torch.max(grad_onehot, -1)

                x_adv = x_adv*(argmax==0).to(x_adv.dtype) + argmax
                x_adv = x_adv.detach()

        # resume context
        if self_training_context == True:
            self.train()
        else:
            self.eval()

        return x_adv.detach()

    """
    def get_adv_hotflip(self, x, y, syn_valid, text_like_syn, attack_type_dict):
        
        # record context
        self_training_context = self.training
        # set context
        if self.eval_adv_mode:
            self.eval()
        else:
            self.train()

        device = x.device
        # get param of attacks

        num_steps=attack_type_dict['num_steps']
        loss_func=attack_type_dict['loss_func']

        batch_size, text_len = x.shape

        onehot_input = self.get_onehot_from_input(x) #bs, len, len_of_vocab
        onehot_mask = self.get_onehot_mask_from_syn(syn_valid, text_like_syn)
        #9.25 04:09
        #embd = self.get_embd_from_onehot(onehot_input)
        embd = self.text_to_embd(x)
        logit = self.embd_to_logit(embd)

        x_adv = torch.zeros_like(x)
        x_adv = x.detach()

        onehot_input_adv = torch.zeros_like(onehot_input)
        onehot_input_adv = onehot_input.detach()

        for i in range(num_steps):
            onehot_input_adv.requires_grad_()
            with torch.enable_grad():
                embd_adv = self.get_embd_from_onehot(onehot_input_adv)
                logit_adv = self.embd_to_logit(embd_adv)
                if loss_func=='ce':
                    loss = F.cross_entropy(logit_adv, y, reduction='sum')
                elif loss_func=='kl':
                    criterion_kl = nn.KLDivLoss(reduction="sum")
                    loss = criterion_kl(F.log_softmax(logit_adv, dim=1), F.softmax(logit, dim=1))

                grad = torch.autograd.grad(loss, [onehot_input_adv])[0] * (onehot_mask)

            _, argmax = torch.max(grad, -1)

            #modify onehot_input_adv 

            temp = -1*onehot_input_adv*(argmax!=0).to(torch.float).unsqueeze(-1) + onehot_input_adv
            onehot_input_adv = (temp + self.get_onehot_from_input(argmax)*(argmax!=0).to(torch.float).unsqueeze(-1)).detach()

        # resume context
        if self_training_context == True:
            self.train()
        else:
            self.eval()

        _, out = torch.max(onehot_input_adv,-1)

        return out.detach()
    """

    def get_embd_adv(self, embd, y, attack_type_dict):
        
        # record context
        self_training_context = self.training
        # set context
        if self.eval_adv_mode:
            self.eval()
        else:
            self.train()

        device = embd.device
        # get param of attacks

        num_steps=attack_type_dict['num_steps']
        step_size=attack_type_dict['step_size']
        random_start=attack_type_dict['random_start']
        epsilon=attack_type_dict['epsilon']
        loss_func=attack_type_dict['loss_func']
        direction=attack_type_dict['direction']
        ball_range=attack_type_dict['ball_range']

        batch_size=len(embd)

        embd_ori = embd
        
        # random start
        if random_start:
            embd_adv = embd_ori.detach() + 0.001 * torch.randn(embd_ori.shape).to(device).detach()
        else:
            embd_adv = embd_ori.detach()


        for _ in range(num_steps):
            embd_adv.requires_grad_()
            grad = 0
            with torch.enable_grad():
                if loss_func=='ce':
                    logit_adv = self.embd_to_logit(embd_adv)
                    if direction == "towards":
                        loss = -F.cross_entropy(logit_adv, y, reduction='sum')
                    elif direction == "away":
                        loss = F.cross_entropy(logit_adv, y, reduction='sum')
                elif loss_func=='kl':
                    logit_adv = self.embd_to_logit(embd_adv)
                    logit_ori = self.embd_to_logit(embd_ori)
                    assert(direction == "away")

                    criterion_kl = nn.KLDivLoss(reduction="sum")
                    loss = criterion_kl(F.log_softmax(logit_adv, dim=1),
                                        F.softmax(logit_ori, dim=1))

                grad = torch.autograd.grad(loss, [embd_adv])[0]

            if ball_range=='sentence':
                grad=self.l2_project_sent(grad)
                embd_adv = embd_adv.detach() + step_size * grad.detach()
                perturbation = self.l2_clip_sent(embd_adv-embd_ori, epsilon)
            elif ball_range=='word':
                grad=self.l2_project(grad)
                embd_adv = embd_adv.detach() + step_size * grad.detach()
                perturbation = self.l2_clip(embd_adv-embd_ori, epsilon)
            else:
                assert NotImplementedError

            embd_adv = embd_ori.detach() + perturbation.detach()
            
        # resume context
        if self_training_context == True:
            self.train()
        else:
            self.eval()

        class bpda_get_embd_adv(torch.autograd.Function):
            
            @staticmethod
            def forward(ctx, embd):
                return embd_adv.detach()
            @staticmethod
            def backward(ctx, grad_output):
                # BPDA, approximate gradients
                return grad_output

        return embd_adv.detach()
        #return bpda_get_embd_adv.apply(embd)


    def embd_to_embdnew(self,embd):
        embdnew = self.linear_transform_embd(embd)
        return embdnew

    def embd_to_text(self, embd):
        bs, sen_len, embd_dim = embd.shape
        text = embd.reshape(-1, embd_dim).mm(self.inverse_embedding.weight)
        return text.reshape(bs, sen_len, -1)
        
    def embdnew_to_embd(self, embdnew):
        embd = self.inverse_linear_transform_embd(embdnew)
        return embd

    def loss_text_adv(self, input, label):
        #p = -F.log_softmax(input, dim=-1)
        #loss = p*(label.to(p.dtype)).sum()
        input_shape = input.shape
        return F.cross_entropy(input.reshape(-1,input_shape[-1]), label.reshape(-1) )

    def text_to_radius(self, inp):
        saved_radius = self.word_synonym_radius(inp) # n, len, 1
        return saved_radius.detach()

    def text_to_embd(self, inp):
        x = self.embedding(inp)
        if self.opt.embd_transform:
            x = self.linear_transform_embd_1(x)
            #x = F.relu(x)
            #x = self.linear_transform_embd_2(x)
            #x = F.relu(x)
            #x = self.linear_transform_embd_3(x)
        return x

    def forward(self, mode, input, comb_p = None, label=None, text_like_syn_embd=None, text_like_syn_valid=None, text_like_syn=None, attack_type_dict=None, text_for_vis=None, record_for_vis=None):
        if mode == "get_embd_adv":
            assert(attack_type_dict is not None)
            out = self.get_embd_adv(input, label, attack_type_dict)
        if mode == "get_adv_by_convex_syn":
            assert(attack_type_dict is not None)
            assert(text_like_syn_embd is not None)
            assert(text_like_syn_valid is not None)
            out = self.get_adv_by_convex_syn(input, label, text_like_syn_embd, text_like_syn_valid, text_like_syn, attack_type_dict, text_for_vis, record_for_vis)
        if mode == "embd_to_logit":
            out = self.embd_to_logit(input)
        if mode == "text_to_embd":
            out = self.text_to_embd(input)
        if mode == "text_to_radius":
            out = self.text_to_radius(input)
        if mode == "text_to_logit":
            embd = self.text_to_embd(input)
            #embd = self.embd_to_embdnew(embd)
            out = self.embd_to_logit(embd)
        if mode == "text_syn_p_to_logit":
            assert(comb_p is not None)
            bs, tl, sl = input.shape
            text_like_syn_embd = self.text_to_embd(input.reshape(bs, tl*sl)).reshape(bs, tl, sl, -1)
            embd = (comb_p*text_like_syn_embd).sum(-2)
            out = self.embd_to_logit(embd)
        if mode == "embd_to_embdnew":
            out = self.embd_to_embdnew(input)
        if mode == "embd_to_text":
            out = self.embd_to_text(input)
        if mode == "embdnew_to_embd":
            out = self.embdnew_to_embd(input)
        if mode == "update_inverse_linear_transform_embd":
            self.update_inverse_linear_transform_embd()
            out = None
        if mode == "update_linear_transform_embd":
            self.update_linear_transform_embd()
            out = None
        if mode == "loss_text_adv":
            out = self.loss_text_adv(input, label)
        if mode == "loss_radius":
            out = self.loss_radius(input, label)

        if mode == "get_adv_hotflip":
            assert(attack_type_dict is not None)
            assert(text_like_syn is not None)
            assert(text_like_syn_valid is not None)
            out = self.get_adv_hotflip(input, label, text_like_syn_valid, text_like_syn, attack_type_dict)


        return out




""" ori
    def get_embd_adv(self, embd, y, attack_type_dict):
        
        # record context
        self_training_context = self.training
        # set context
        self.eval()

        device = embd.device
        # get param of attacks

        num_steps=attack_type_dict['num_steps']
        step_size=attack_type_dict['step_size']
        random_start=attack_type_dict['random_start']
        epsilon=attack_type_dict['epsilon']
        loss_func=attack_type_dict['loss_func']
        direction=attack_type_dict['direction']

        batch_size=len(embd)

        embd_ori = embd
        
        # random start
        if random_start:
            embd_adv = embd_ori.detach() + 0.001 * torch.randn(embd_ori.shape).to(device).detach()
        else:
            embd_adv = embd_ori.detach()


        for _ in range(num_steps):
            embd_adv.requires_grad_()
            grad = 0
            with torch.enable_grad():
                if loss_func=='ce':
                    logit_adv = self.embd_to_logit(embd_adv)
                    if direction == "towards":
                        loss = -F.cross_entropy(logit_adv, y, reduction='sum')
                    elif direction == "away":
                        loss = F.cross_entropy(logit_adv, y, reduction='sum')
                grad = torch.autograd.grad(loss, [embd_adv])[0]

            grad=self.l2_project(grad)

            embd_adv = embd_adv.detach() + step_size * grad.detach()
            
            perturbation = self.l2_clip(embd_adv-embd_ori, epsilon)
            embd_adv = embd_ori.detach() + perturbation.detach()
            
        # resume context
        if self_training_context == True:
            self.train()
        else:
            self.eval()

        return embd_adv.detach()
"""