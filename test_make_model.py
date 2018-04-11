import mxnet as mx
import numpy as np
from config import config as cfg
import os


import sys, time, logging
from angleIter import angleIter
#from make_symbol import gen_symbol, gen_model, feval_l1_angleMetric, feval_acc_angleMetric
from make_model import gen_feat_model
from tool import mytick,viz_score_predict, viz_bbox_gdt, viz_target
import matplotlib.pyplot as plt

#it=angleIter(symbol_mod)#'../../data/angleList.txt','../../data/img',1,100)
#d0=it.next()
d = mx.nd.random.uniform(0, 255,shape=(1,3,448,448), ctx=cfg.train.ctx)
logging.info('PID:%d'%os.getpid())
feat_model, feat_trainer = gen_feat_model()
y= feat_model(d)
print y.shape
print y[0][0]
