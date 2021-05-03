# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import os
import math


class H_Discriminator(nn.Module):
    def __init__(self, fea_dim):
        super(H_Discriminator, self).__init__()

        self.function = nn.Sequential(nn.Linear(fea_dim*2,256),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,2))
        
    def forward(self, a, b, dataset_id=None, iters=1):

        bs, fea_dim =  a.shape
        lsoftmax = nn.LogSoftmax(1)

        a_expand = a.detach().reshape(bs,1,fea_dim).repeat(1,bs,1).reshape(bs*bs,fea_dim)
        b_expand = b.detach().reshape(1,bs,fea_dim).repeat(bs,1,1).reshape(bs*bs,fea_dim)

        params = [param for param in self.function.parameters() if param.requires_grad]
        optimizer = torch.optim.Adam(params, lr=2e-5,weight_decay=2e-5)

        for i in range(iters):
            optimizer.zero_grad()
            logits = self.function(torch.cat((a.detach(),b.detach()), 1))
            logits = lsoftmax(logits)
            logits_cross = self.function( torch.cat((a_expand.detach(),b_expand.detach()), 1) )
            logits_cross = lsoftmax(logits_cross).reshape(bs,bs,2)
            loss_for_function = logits[:,0].sum()/bs + math.log(bs-1) +  (logits_cross[:,:,1].sum() - torch.diag(logits_cross[:,:,1]).sum()) /bs
            loss_for_function *= -1
            loss_for_function.backward()
            optimizer.step()

    
        logits_final = self.function(torch.cat((a,b), 1))
        logits_final = lsoftmax(logits_final)
        loss = logits_final[:,0].sum()/bs + math.log(bs-1)
        loss *= -1

        return loss


class H_CosCritic(nn.Module):
    def __init__(self, fea_dim):
        super(H_CosCritic, self).__init__()

        self.proj_a = nn.Sequential(nn.Linear(fea_dim,512),nn.ReLU(),nn.Linear(512,256))
        self.proj_b = nn.Sequential(nn.Linear(fea_dim,512),nn.ReLU(),nn.Linear(512,256))
        self.temperature = 0.2

    def forward(self, a, b, dataset_id=None):
        a = self.proj_a(a)
        a = torch.nn.functional.normalize(a)
        b = self.proj_b(b)
        b = torch.nn.functional.normalize(b)

        bs, fea_dim = a.shape
        f_a_b = torch.mm(a, torch.transpose(b, 0, 1) )#bs*768 * 768*bs = bs * bs
        f_a_b = f_a_b/self.temperature

        lsoftmax = nn.LogSoftmax(1)
        log_p_f_a_b = lsoftmax(f_a_b)

        out = torch.zeros(bs, 2)
        out[:, 0] = torch.diag(log_p_f_a_b)

        logit_pos = (a*b).sum(-1)

        return 
