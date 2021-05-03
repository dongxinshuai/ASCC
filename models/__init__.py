# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import numpy as np

from .CNNBasic import BasicCNN1D,BasicCNN2D, AdvBasicCNN1D, AdvBasicCNN2D
from .LSTMBI import LSTMBI, AdvLSTMBI
from .ForSnli import AdvDecAtt, AdvDecAtt_FromCert, AdvBOW, AdvEntailmentCNN

def setup(opt):
    
    if opt.model == 'cnn_adv':
        model = AdvBasicCNN1D(opt)
    elif opt.model == 'decomp_att_adv':
        model = AdvDecAtt_FromCert(opt)
    elif opt.model == 'bow_adv':
        model = AdvBOW(opt)
    elif opt.model ==  'bilstm_adv':
        model = AdvLSTMBI(opt)
    else:
        raise Exception("model not supported: {}".format(opt.model))
    return model
