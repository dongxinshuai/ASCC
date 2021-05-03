# -*- coding: utf-8 -*-
import torch 
import numpy as np
from torch import nn
import torch.nn.functional as F
from .BaseModel import BaseModel
from .BaseModelAdv import AdvBaseModel


class Dev_Att_Encoder(nn.Module):

    def __init__(self, embedding_size, hidden_size=300, para_init=0.01):
        super(Dev_Att_Encoder, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.para_init = para_init

        self.input_linear = nn.Linear(
            self.embedding_size, self.hidden_size, bias=False)  # linear transformation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, self.para_init)
                # m.bias.data.uniform_(-0.01, 0.01)

    def forward(self, embd_1, embd_2):

        batch_size, sent_len, embd_size = embd_1.shape

        embd_1 = embd_1.view(-1, embd_size)
        embd_2 = embd_2.view(-1, embd_size)

        sent1_linear = self.input_linear(embd_1).view(
            batch_size, sent_len, self.hidden_size)
        sent2_linear = self.input_linear(embd_2).view(
            batch_size, sent_len, self.hidden_size)

        return sent1_linear, sent2_linear

class Dev_Att_Atten(nn.Module):
    '''
        intra sentence attention
    '''

    def __init__(self, label_size, hidden_size=300, para_init=0.01):
        super(Dev_Att_Atten, self).__init__()

        self.hidden_size = hidden_size
        self.label_size = label_size
        self.para_init = para_init

        self.mlp_f = self._mlp_layers(self.hidden_size, self.hidden_size)
        self.mlp_g = self._mlp_layers(2 * self.hidden_size, self.hidden_size)
        self.mlp_h = self._mlp_layers(2 * self.hidden_size, self.hidden_size)

        self.final_linear = nn.Linear(
            self.hidden_size, self.label_size, bias=True)

        self.log_prob = nn.LogSoftmax()

        '''initialize parameters'''
        for m in self.modules():
            # print m
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, self.para_init)
                m.bias.data.normal_(0, self.para_init)

    def _mlp_layers(self, input_dim, output_dim):
        mlp_layers = []
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.Linear(
            input_dim, output_dim, bias=True))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.Linear(
            output_dim, output_dim, bias=True))
        mlp_layers.append(nn.ReLU())        
        return nn.Sequential(*mlp_layers)   # * used to unpack list

    def forward(self, sent1_linear, sent2_linear):
        '''
            sent_linear: batch_size x length x hidden_size
        '''
        len1 = sent1_linear.size(1)
        len2 = sent2_linear.size(1)

        '''attend'''

        f1 = self.mlp_f(sent1_linear.view(-1, self.hidden_size))
        f2 = self.mlp_f(sent2_linear.view(-1, self.hidden_size))

        f1 = f1.view(-1, len1, self.hidden_size)
        # batch_size x len1 x hidden_size
        f2 = f2.view(-1, len2, self.hidden_size)
        # batch_size x len2 x hidden_size

        score1 = torch.bmm(f1, torch.transpose(f2, 1, 2))
        # e_{ij} batch_size x len1 x len2
        prob1 = F.softmax(score1.view(-1, len2)).view(-1, len1, len2)
        # batch_size x len1 x len2

        score2 = torch.transpose(score1.contiguous(), 1, 2)
        score2 = score2.contiguous()
        # e_{ji} batch_size x len2 x len1
        prob2 = F.softmax(score2.view(-1, len1)).view(-1, len2, len1)
        # batch_size x len2 x len1

        sent1_combine = torch.cat(
            (sent1_linear, torch.bmm(prob1, sent2_linear)), 2)
        # batch_size x len1 x (hidden_size x 2)
        sent2_combine = torch.cat(
            (sent2_linear, torch.bmm(prob2, sent1_linear)), 2)
        # batch_size x len2 x (hidden_size x 2)

        '''sum'''
        g1 = self.mlp_g(sent1_combine.view(-1, 2 * self.hidden_size))
        g2 = self.mlp_g(sent2_combine.view(-1, 2 * self.hidden_size))
        g1 = g1.view(-1, len1, self.hidden_size)
        # batch_size x len1 x hidden_size
        g2 = g2.view(-1, len2, self.hidden_size)
        # batch_size x len2 x hidden_size

        sent1_output = torch.sum(g1, 1)  # batch_size x 1 x hidden_size
        sent1_output = torch.squeeze(sent1_output, 1)
        sent2_output = torch.sum(g2, 1)  # batch_size x 1 x hidden_size
        sent2_output = torch.squeeze(sent2_output, 1)

        input_combine = torch.cat((sent1_output, sent2_output), 1)
        # batch_size x (2 * hidden_size)
        h = self.mlp_h(input_combine)
        # batch_size * hidden_size

        # if sample_id == 15:
        #     print '-2 layer'
        #     print h.data[:, 100:150]

        logits = self.final_linear(h)
        #log_prob = self.log_prob(logits)
        return logits

class AdvDecAtt(AdvBaseModel): 
    def __init__(self, opt ):
        super(AdvDecAtt, self).__init__(opt)

        self.encoder = Dev_Att_Encoder(self.embedding_out_dim)
        self.atten = Dev_Att_Atten(self.label_size)

    def embd_to_logit(self, embd_p, embd_h, x_p_mask, x_h_mask):
        p_linear, h_linear = self.encoder(embd_p, embd_h)
        logits = self.atten(p_linear, h_linear)
        return logits

    def text_to_embd(self, x_p, x_h):
        embd_x_p = self.embedding(x_p)
        embd_x_h = self.embedding(x_h)
        if self.opt.embd_transform:
            embd_x_p = self.linear_transform_embd_1(embd_x_p)
            embd_x_h = self.linear_transform_embd_1(embd_x_h)
            #embd_x_p = F.relu(embd_x_p)
            #x = self.linear_transform_embd_2(x)
            #embd_x_h = F.relu(embd_x_h)
            #x = self.linear_transform_embd_3(x)
        return embd_x_p, embd_x_h

    def get_adv_by_convex_syn(self, embd_p, embd_h, y, x_p_text_like_syn, x_p_syn_embd, x_p_syn_valid, x_h_text_like_syn, x_h_syn_embd, x_h_syn_valid, x_p_mask, x_h_mask, attack_type_dict):
        
        #noted that if attack hypo only then the output x_p_comb_p is meaningless

        # record context
        self_training_context = self.training
        # set context
        if self.eval_adv_mode:
            self.eval()
        else:
            self.train()

        device = embd_p.device
        # get param of attacks

        num_steps=attack_type_dict['num_steps']
        loss_func=attack_type_dict['loss_func']
        w_optm_lr=attack_type_dict['w_optm_lr']
        sparse_weight = attack_type_dict['sparse_weight']
        out_type = attack_type_dict['out_type']
        attack_hypo_only = attack_type_dict['attack_hypo_only'] if 'attack_hypo_only' in attack_type_dict else True

        batch_size, text_len, embd_dim = embd_p.shape
        batch_size, text_len, syn_num, embd_dim = x_p_syn_embd.shape

        w_p = torch.empty(batch_size, text_len, syn_num, 1).to(device).to(embd_p.dtype)
        w_h = torch.empty(batch_size, text_len, syn_num, 1).to(device).to(embd_p.dtype)
        #ww = torch.zeros(batch_size, text_len, syn_num, 1).to(device).to(embd.dtype)
        #ww = ww+500*(syn_valid.reshape(batch_size, text_len, syn_num, 1)-1)
        nn.init.kaiming_normal_(w_p)
        nn.init.kaiming_normal_(w_h)
        w_p.requires_grad_()
        w_h.requires_grad_()
        
        import utils
        params = [w_p, w_h] 
        optimizer = utils.getOptimizer(params,name='adam', lr=w_optm_lr,weight_decay=2e-5)

        def get_comb_p(w, syn_valid):
            ww=w*syn_valid.reshape(batch_size, text_len, syn_num, 1) + 10000*(syn_valid.reshape(batch_size, text_len, syn_num, 1)-1)
            return F.softmax(ww, -2)

        def get_comb_ww(w, syn_valid):
            ww=w*syn_valid.reshape(batch_size, text_len, syn_num, 1) + 10000*(syn_valid.reshape(batch_size, text_len, syn_num, 1)-1)
            return ww

        def get_comb(p, syn):
            return (p* syn.detach()).sum(-2)


        embd_p_ori=embd_p.detach()
        embd_h_ori=embd_h.detach()
        logit_ori = self.embd_to_logit(embd_p_ori, embd_h_ori, x_p_mask, x_h_mask)

        for _ in range(num_steps):
            optimizer.zero_grad()
            with torch.enable_grad():
                ww_p = get_comb_ww(w_p, x_p_syn_valid)
                ww_h = get_comb_ww(w_h, x_h_syn_valid)
                #comb_p = get_comb_p(w, syn_valid)
                embd_p_adv = get_comb(F.softmax(ww_p, -2), x_p_syn_embd)
                embd_h_adv = get_comb(F.softmax(ww_h, -2), x_h_syn_embd)
                if attack_hypo_only:
                    logit_adv = self.embd_to_logit(embd_p_ori, embd_h_adv, x_p_mask, x_h_mask)
                else:
                    logit_adv = self.embd_to_logit(embd_p_adv, embd_h_adv, x_p_mask, x_h_mask)

                if loss_func=='ce':
                    loss = -F.cross_entropy(logit_adv, y, reduction='sum')
                elif loss_func=='kl':
                    criterion_kl = nn.KLDivLoss(reduction="sum")
                    loss = -criterion_kl(F.log_softmax(logit_adv, dim=1),
                                        F.softmax(logit_ori.detach(), dim=1))

                #print("ad loss:", loss.data.item())
                                    
                if sparse_weight !=0:
                    #loss_sparse = (comb_p*comb_p).mean()
                    if attack_hypo_only:
                        loss_sparse = (-F.softmax(ww_h, -2)*F.log_softmax(ww_h, -2)).sum(-2).mean() 
                    else:
                        loss_sparse = ( (-F.softmax(ww_p, -2)*F.log_softmax(ww_p, -2)).sum(-2).mean() + (-F.softmax(ww_h, -2)*F.log_softmax(ww_h, -2)).sum(-2).mean() )/2
                    #loss -= sparse_weight*loss_sparse
                    
                    loss = loss + sparse_weight*loss_sparse
                    #print(loss_sparse.data.item())

            #loss*=1000
            loss.backward()
            optimizer.step()

        #print((ww-w).max())

        x_p_comb_p = get_comb_p(w_p, x_p_syn_valid)
        x_h_comb_p = get_comb_p(w_h, x_h_syn_valid)

        """
        out = get_comb(comb_p, syn)
        delta = (out-embd_ori).reshape(batch_size*text_len,embd_dim)
        delta = F.pairwise_distance(delta, torch.zeros_like(delta), p=2.0)
        valid = (delta>0.01).to(device).to(delta.dtype)
        delta = (valid*delta).sum()/valid.sum()
        print("mean l2 dis between embd and embd_adv:", delta.data.item())
        #print("mean max comb_p:", (comb_p.max(-2)[0]).mean().data.item())
        """

        # resume context
        if self_training_context == True:
            self.train()
        else:
            self.eval()

        if out_type == "comb_p":
            return x_p_comb_p.detach(), x_h_comb_p.detach()
        elif out_type == "text":
            assert(x_p_text_like_syn is not None) # n l synlen
            assert(x_h_text_like_syn is not None) # n l synlen
            x_p_comb_p = x_p_comb_p.reshape(batch_size* text_len, syn_num)
            x_h_comb_p = x_h_comb_p.reshape(batch_size* text_len, syn_num)
            ind_x_p = x_p_comb_p.max(-1)[1] # shape batch_size* text_len
            ind_x_h = x_h_comb_p.max(-1)[1] # shape batch_size* text_len
            adv_text_x_p = (x_p_text_like_syn.reshape(batch_size* text_len, syn_num)[np.arange(batch_size*text_len), ind_x_p]).reshape(batch_size, text_len)
            adv_text_x_h = (x_h_text_like_syn.reshape(batch_size* text_len, syn_num)[np.arange(batch_size*text_len), ind_x_h]).reshape(batch_size, text_len)
            return adv_text_x_p, adv_text_x_h


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

    def get_adv_hotflip(self, x_p, x_h, y, x_p_text_like_syn, x_p_syn_valid, x_h_text_like_syn, x_h_syn_valid, x_p_mask, x_h_mask, attack_type_dict):
        
        # record context
        self_training_context = self.training
        # set context
        if self.eval_adv_mode:
            self.eval()
        else:
            self.train()

        device = x_p.device
        # get param of attacks

        num_steps=attack_type_dict['num_steps']
        loss_func=attack_type_dict['loss_func']

        batch_size, text_len = x_p.shape

        onehot_mask_h = self.get_onehot_mask_from_syn(x_h_syn_valid, x_h_text_like_syn)

        embd_p, embd_h = self.text_to_embd(x_p, x_h)
        logit = self.embd_to_logit(embd_p, embd_h, x_p_mask, x_h_mask)

        x_p_adv = torch.zeros_like(x_p)
        x_p_adv = x_p.detach()

        x_h_adv = torch.zeros_like(x_h)
        x_h_adv = x_h.detach()

        for i in range(num_steps):
            embd_p_adv, embd_h_adv= self.text_to_embd(x_p_adv, x_h_adv)
            embd_h_adv.requires_grad_()
            with torch.enable_grad():

                # attack_hypo_only:
                logit_adv = self.embd_to_logit(embd_p_adv, embd_h_adv, x_p_mask, x_h_mask)

                if loss_func=='ce':
                    loss = F.cross_entropy(logit_adv, y, reduction='sum')
                elif loss_func=='kl':
                    criterion_kl = nn.KLDivLoss(reduction="sum")
                    loss = criterion_kl(F.log_softmax(logit_adv, dim=1), F.softmax(logit, dim=1))

                grad_embd_h = torch.autograd.grad(loss, [embd_h_adv])[0]

            with torch.no_grad():
                if self.opt.embd_transform:
                    grad_onehot_h = torch.mm( grad_embd_h.reshape(batch_size*text_len,-1), torch.mm(self.embedding.weight, self.linear_transform_embd_1.weight.permute(1,0)).permute(1,0) )
                else:
                    grad_onehot_h = torch.mm( grad_embd_h.reshape(batch_size*text_len,-1), self.embedding.weight.permute(1,0) )
                grad_onehot_h =  grad_onehot_h.reshape(batch_size, text_len, -1) * (onehot_mask_h)
                _, argmax = torch.max(grad_onehot_h, -1)

                x_h_adv = x_h_adv*(argmax==0).to(x_h_adv.dtype) + argmax
                x_h_adv = x_h_adv.detach()

        # resume context
        if self_training_context == True:
            self.train()
        else:
            self.eval()

        return x_p_adv.detach(), x_h_adv.detach()





    def get_embd_adv(self, embd_p, embd_h, y, x_p_mask, x_h_mask, attack_type_dict):

        # record context
        self_training_context = self.training
        # set context
        if self.eval_adv_mode:
            self.eval()
        else:
            self.train()

        device = embd_p.device
        # get param of attacks

        num_steps=attack_type_dict['num_steps']
        step_size=attack_type_dict['step_size']
        random_start=attack_type_dict['random_start']
        epsilon=attack_type_dict['epsilon']
        loss_func=attack_type_dict['loss_func']
        direction=attack_type_dict['direction']
        ball_range=attack_type_dict['ball_range']
        attack_hypo_only=attack_type_dict['attack_hypo_only']
        assert(attack_hypo_only)

        batch_size=len(embd_p)

        embd_p_ori = embd_p
        embd_h_ori = embd_h
        
        # random start
        if random_start:
            embd_p_adv = embd_p_ori.detach() + 0.001 * torch.randn(embd_p_ori.shape).to(device).detach()
            embd_h_adv = embd_h_ori.detach() + 0.001 * torch.randn(embd_h_ori.shape).to(device).detach()
        else:
            embd_p_adv = embd_p_ori.detach()
            embd_h_adv = embd_h_ori.detach()

        for _ in range(num_steps):
            embd_h_adv.requires_grad_()
            grad = 0
            with torch.enable_grad():
                if loss_func=='ce':
                    logit_adv = self.embd_to_logit(embd_p_ori, embd_h_adv, x_p_mask, x_h_mask)
                    if direction == "towards":
                        loss = -F.cross_entropy(logit_adv, y, reduction='sum')
                    elif direction == "away":
                        loss = F.cross_entropy(logit_adv, y, reduction='sum')
                elif loss_func=='kl':
                    logit_adv = self.embd_to_logit(embd_p_ori, embd_h_adv, x_p_mask, x_h_mask)
                    logit_ori = self.embd_to_logit(embd_p_ori, embd_h_ori, x_p_mask, x_h_mask)
                    assert(direction == "away")

                    criterion_kl = nn.KLDivLoss(reduction="sum")
                    loss = criterion_kl(F.log_softmax(logit_adv, dim=1),
                                        F.softmax(logit_ori, dim=1))

                grad = torch.autograd.grad(loss, [embd_h_adv])[0]

            if ball_range=='sentence':
                grad=self.l2_project_sent(grad)
                embd_h_adv = embd_h_adv.detach() + step_size * grad.detach()
                perturbation = self.l2_clip_sent(embd_h_adv-embd_h_ori, epsilon)
            elif ball_range=='word':
                grad=self.l2_project(grad)
                embd_h_adv = embd_h_adv.detach() + step_size * grad.detach()
                perturbation = self.l2_clip(embd_h_adv-embd_h_ori, epsilon)
            else:
                assert NotImplementedError

            embd_h_adv = embd_h_ori.detach() + perturbation.detach()
            
        # resume context
        if self_training_context == True:
            self.train()
        else:
            self.eval()

        class bpda_get_embd_adv(torch.autograd.Function):
            
            @staticmethod
            def forward(ctx, embd_p, embd_h):
                return embd_p_ori.detach(), embd_h_adv.detach()
            @staticmethod
            def backward(ctx, grad_output):
                # BPDA, approximate gradients
                return grad_output

        return embd_p_ori.detach(), embd_h_adv.detach()
        #return bpda_get_embd_adv.apply(embd)



    def forward(self, mode, x_p, x_h, x_p_comb_p=None, x_h_comb_p=None, label=None, x_p_text_like_syn=None, x_p_text_like_syn_embd=None, x_p_text_like_syn_valid=None, x_h_text_like_syn=None, x_h_text_like_syn_embd=None, x_h_text_like_syn_valid=None, x_p_mask=None, x_h_mask=None, attack_type_dict=None):
        if mode == "get_adv_by_convex_syn":
            assert(attack_type_dict is not None)
            assert(x_p_text_like_syn_embd is not None)
            assert(x_p_text_like_syn_valid is not None)
            assert(x_h_text_like_syn_embd is not None)
            assert(x_h_text_like_syn_valid is not None)

            out = self.get_adv_by_convex_syn(x_p, x_h, label, x_p_text_like_syn, x_p_text_like_syn_embd, x_p_text_like_syn_valid, x_h_text_like_syn, x_h_text_like_syn_embd, x_h_text_like_syn_valid, x_p_mask, x_h_mask, attack_type_dict)
        if mode == "embd_to_logit":
            out = self.embd_to_logit(x_p, x_h, x_p_mask, x_h_mask)
        if mode == "text_to_embd":
            out = self.text_to_embd(x_p, x_h)
        if mode == "text_to_logit":
            embd_p, embd_h = self.text_to_embd(x_p, x_h)
            out = self.embd_to_logit(embd_p, embd_h, x_p_mask, x_h_mask)
        if mode == "text_syn_p_to_logit":
            assert(x_p_comb_p is not None)
            assert(x_h_comb_p is not None)
            bs, tl, sl = x_p.shape
            x_p_text_like_syn_embd, x_h_text_like_syn_embd = self.text_to_embd(x_p.reshape(bs, tl*sl), x_h.reshape(bs, tl*sl))
            x_p_text_like_syn_embd=x_p_text_like_syn_embd.reshape(bs, tl, sl, -1)
            x_h_text_like_syn_embd=x_h_text_like_syn_embd.reshape(bs, tl, sl, -1)
            embd_p = (x_p_comb_p*x_p_text_like_syn_embd).sum(-2)
            embd_h = (x_h_comb_p*x_h_text_like_syn_embd).sum(-2)
            out = self.embd_to_logit(embd_p, embd_h, x_p_mask, x_h_mask)
        if mode == "text_syn_p_to_logit_hypo_only":
            assert(x_h_comb_p is not None)
            bs, tl, sl = x_h.shape
            embd_p, x_h_text_like_syn_embd = self.text_to_embd(x_p, x_h.reshape(bs, tl*sl))
            x_h_text_like_syn_embd=x_h_text_like_syn_embd.reshape(bs, tl, sl, -1)
            embd_h = (x_h_comb_p*x_h_text_like_syn_embd).sum(-2)
            out = self.embd_to_logit(embd_p, embd_h, x_p_mask, x_h_mask)

        if mode == "get_embd_adv":
            assert(attack_type_dict is not None)
            out = self.get_embd_adv(x_p, x_h, label, x_p_mask, x_h_mask, attack_type_dict)

        if mode == "get_adv_hotflip":
            out = self.get_adv_hotflip(x_p, x_h, label, x_p_text_like_syn, x_p_text_like_syn_valid, x_h_text_like_syn, x_h_text_like_syn_valid, x_p_mask, x_h_mask, attack_type_dict)
        
        return out



class AdvDecAtt_FromCert(AdvDecAtt): 
    def __init__(self, opt ):
        super(AdvDecAtt_FromCert, self).__init__(opt)

        def get_feedforward_layers(num_layers, input_size, hidden_size, output_size):
            layers = []
            for i in range(num_layers):
                layer_in_size = input_size if i == 0 else hidden_size
                layer_out_size = output_size if i == num_layers - 1 else hidden_size
                layers.append(nn.Dropout(self.opt.keep_dropout))
                layers.append(nn.Linear(layer_in_size, layer_out_size))
                if i < num_layers - 1:
                    layers.append(nn.ReLU(True))
            return layers

        num_layers=2
        hidden_size=self.embedding_out_dim

        ff_layers = get_feedforward_layers(num_layers, self.embedding_out_dim, hidden_size, 1)
        self.feedforward = nn.Sequential(*ff_layers)

        compare_layers = get_feedforward_layers(num_layers, 2 * self.embedding_out_dim, hidden_size, hidden_size)
        self.compare_ff = nn.Sequential(*compare_layers)

        output_layers = get_feedforward_layers(num_layers, 2 * hidden_size, hidden_size, hidden_size)
        output_layers.append(nn.Linear(hidden_size, self.label_size))
        output_layers.append(nn.LogSoftmax(dim=-1))
        self.output_layer = nn.Sequential(*output_layers)

    def embd_to_logit(self, embd_p, embd_h, x_p_mask, x_h_mask):
        """
        Forward pass of DecompAttentionModel.
        Args:
        batch: A batch dict from an EntailmentDataset with the following keys:
            - prem: tensor of word vector indices for premise (B, p, 1)
            - hypo: tensor of word vector indices for hypothesis (B, h, 1)
            - prem_mask: binary mask over premise words (1 for real, 0 for pad), size (B, p)
            - hypo_mask: binary mask over hypothesis words (1 for real, 0 for pad), size (B, h)
            - prem_lengths: lengths of premises, size (B,)
            - hypo_lengths: lengths of hypotheses, size (B,)
        compute_bounds: If True compute the interval bounds and reutrn an IntervalBoundedTensor as logits. Otherwise just use the values
        cert_eps: float, scaling factor for the interval bounds.
        """

        embd_p = embd_p*x_p_mask.unsqueeze(-1)
        embd_h = embd_h*x_h_mask.unsqueeze(-1)

        prem_weights = self.feedforward(embd_p)*x_p_mask.unsqueeze(-1) # (bXpX1)
        hypo_weights = self.feedforward(embd_h)*x_h_mask.unsqueeze(-1) # (bXhX1)

        attention = torch.bmm(prem_weights, hypo_weights.permute(0,2,1)) # (bXpX1) X (bX1Xh) => (bXpXh)
        attention_mask = x_p_mask.unsqueeze(-1)*x_h_mask.unsqueeze(1)
        attention_masked = attention + (1 - attention_mask) * -1e20
        attended_prem = self.attend_on(embd_h, embd_p, attention_masked) # (bXpX2e)
        attended_hypo = self.attend_on(embd_p, embd_h, attention_masked.permute(0,2,1)) # (bXhX2e)
        compared_prem = self.compare_ff(attended_prem)*x_p_mask.unsqueeze(-1)  # (bXpXhid)
        compared_hypo = self.compare_ff(attended_hypo)*x_h_mask.unsqueeze(-1)  # (bXhXhid)
        prem_aggregate = torch.sum(compared_prem, dim=1) # (bXhid)
        hypo_aggregate = torch.sum(compared_hypo, dim=1) # (bXhid)
        aggregate = torch.cat([prem_aggregate, hypo_aggregate], dim=-1) # (bX2hid)
        return self.output_layer(aggregate) # (b, class_num)

    def attend_on(self, source, target, attention):
        """
        Args:
        - source: (bXsXe)
        - target: (bXtXe)
        - attention: (bXtXs)
        """
        attention_logsoftmax = torch.log_softmax(attention, 1)
        attention_normalized = torch.exp(attention_logsoftmax)
        attended_target = torch.matmul(attention_normalized, source) # (bXtXe)
        return torch.cat([target, attended_target], dim=-1)

    def text_to_embd(self, x_p, x_h):
        embd_x_p = self.embedding(x_p)
        embd_x_h = self.embedding(x_h)
        if self.opt.embd_transform:
            embd_x_p = self.linear_transform_embd_1(embd_x_p)
            embd_x_h = self.linear_transform_embd_1(embd_x_h)
            embd_x_p = F.relu(embd_x_p)
            embd_x_h = F.relu(embd_x_h)
            #x = self.linear_transform_embd_2(x)
            #x = F.relu(x)
            #x = self.linear_transform_embd_3(x)
        return embd_x_p, embd_x_h


class AdvBOW(AdvDecAtt): 
    def __init__(self, opt ):
        super(AdvBOW, self).__init__(opt)
        self.sum_drop = nn.Dropout(self.opt.keep_dropout)
        layers = []
        hidden_size = self.embedding_out_dim
        num_layers = 3
        for i in range(num_layers):
            layers.append(nn.Linear(2*hidden_size, 2*hidden_size))
            layers.append(nn.ReLU(True))
            layers.append(nn.Dropout(self.opt.keep_dropout))
        layers.append(nn.Linear(2*hidden_size, self.label_size))
        layers.append(nn.LogSoftmax(dim=-1)) # why?
        self.layers = nn.Sequential(*layers)

    def embd_to_logit(self, embd_p, embd_h, x_p_mask, x_h_mask):
        
        if self.opt.bow_mean:
            embd_p = (embd_p*x_p_mask.unsqueeze(-1)).sum(-2) / ( x_p_mask.unsqueeze(-1).sum(-2) )
            embd_h = (embd_h*x_h_mask.unsqueeze(-1)).sum(-2) / ( x_h_mask.unsqueeze(-1).sum(-2) )
        else:
            embd_p = (embd_p*x_p_mask.unsqueeze(-1)).sum(-2)
            embd_h = (embd_h*x_h_mask.unsqueeze(-1)).sum(-2)

        input_encoded = torch.cat([embd_p, embd_h], -1)
        logits = self.layers(input_encoded)
        return logits

    def text_to_embd(self, x_p, x_h):
        embd_x_p = self.embedding(x_p)
        embd_x_h = self.embedding(x_h)
        if self.opt.embd_transform:
            embd_x_p = self.linear_transform_embd_1(embd_x_p)
            embd_x_h = self.linear_transform_embd_1(embd_x_h)
            embd_x_p = F.relu(embd_x_p)
            embd_x_h = F.relu(embd_x_h)
            #x = self.linear_transform_embd_2(x)
            #x = F.relu(x)
            #x = self.linear_transform_embd_3(x)
        return embd_x_p, embd_x_h


class AdvEntailmentCNN(AdvDecAtt): 
    def __init__(self, opt ):
        super(AdvEntailmentCNN, self).__init__(opt)

        self.content_dim=opt.__dict__.get("content_dim",256)
        self.kernel_size=opt.__dict__.get("kernel_size",3)

        self.content_p_conv = nn.Sequential(
            nn.Conv1d(in_channels = self.embedding_out_dim,
                      out_channels = self.content_dim, #256
                      kernel_size = self.kernel_size), #3
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = (opt.max_seq_len - self.kernel_size + 1)),
            #nn.MeanPool1d(kernel_size = (opt.max_seq_len - self.kernel_size + 1))
#            nn.AdaptiveMaxPool1d()
        )

        self.content_h_conv = nn.Sequential(
            nn.Conv1d(in_channels = self.embedding_out_dim,
                      out_channels = self.content_dim, #256
                      kernel_size = self.kernel_size), #3
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = (opt.max_seq_len - self.kernel_size + 1)),
            #nn.MeanPool1d(kernel_size = (opt.max_seq_len - self.kernel_size + 1))
#            nn.AdaptiveMaxPool1d()
        )
        self.properties.update(
                {"content_dim":self.content_dim,
                 "kernel_size":self.kernel_size,
                })
        self.dropout_p = nn.Dropout(opt.keep_dropout)
        self.dropout_h = nn.Dropout(opt.keep_dropout)
        self.fc = nn.Linear(self.content_dim*2, opt.label_size)

    def text_to_embd(self, x_p, x_h):
        embd_x_p = self.embedding(x_p)
        embd_x_h = self.embedding(x_h)
        if self.opt.embd_transform:
            embd_x_p = self.linear_transform_embd_1(embd_x_p)
            embd_x_h = self.linear_transform_embd_1(embd_x_h)
            embd_x_p = F.relu(embd_x_p)
            embd_x_h = F.relu(embd_x_h)
            #x = self.linear_transform_embd_2(x)
            #x = F.relu(x)
            #x = self.linear_transform_embd_3(x)
        return embd_x_p, embd_x_h


    def embd_to_logit(self, embd_p, embd_h, x_p_mask, x_h_mask):
        content_p = self.content_p_conv(embd_p.permute(0,2,1)) #64x256x1
        reshaped_p = content_p.view(content_p.size(0), -1) #64x256
        reshaped_p = self.dropout_p(reshaped_p)

        content_h = self.content_h_conv(embd_h.permute(0,2,1)) #64x256x1
        reshaped_h = content_h.view(content_h.size(0), -1) #64x256
        reshaped_h = self.dropout_h(reshaped_h)

        logits = self.fc(torch.cat([reshaped_p, reshaped_h], -1)) #64x3
        return logits

import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--hidden_dim', type=int, default=128,
                    help='hidden_dim')   
    
    
    parser.add_argument('--batch_size', type=int, default=64,
                    help='batch_size')
    parser.add_argument('--embedding_dim', type=int, default=300,
                    help='embedding_dim')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                    help='learning_rate')
    parser.add_argument('--grad_clip', type=float, default=1e-1,
                    help='grad_clip')
    parser.add_argument('--model', type=str, default="lstm",
                    help='model name')
    parser.add_argument('--model', type=str, default="lstm",
                    help='model name')


#
    args = parser.parse_args()
    args.embedding_dim=300
    args.vocab_size=10000
    args.kernel_size=3
    args.num_classes=3
    args.content_dim=256
    args.max_seq_len=50
    
#
#    # Check if args are valid
#    assert args.rnn_size > 0, "rnn_size should be greater than 0"


    return args
 
if __name__ == '__main__':    

    opt = parse_opt()
    m = CNNText(opt)
    content = t.autograd.Variable(t.arange(0,3200).view(-1,50)).long()
    o = m(content)
    print(o.size())

