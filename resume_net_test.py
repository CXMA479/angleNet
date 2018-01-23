import mxnet as mx
import numpy as np
import sys, time, logging
from config import config as cfg
from angleIter import angleIter
from make_symbol import gen_symbol, gen_model, feval_l1_angleMetric
from tool import mytick,viz_score_predict, viz_bbox_gdt, viz_target
import matplotlib.pyplot as plt
from nms.nms import mx_nms
import cPickle as cpk

cfg.it.debug.debug=  True

symbol_mod, fixed_param_names = gen_symbol()


it=angleIter(symbol_mod)#'../../data/angleList.txt','../../data/img',1,100)
d=it.next()


context_file='../data/debug/Sat-Dec-16-16:28:02-2017.cpk'
with open(context_file,'r') as f:
    d=cpk.load(f)

it_anchor=d['anchor']
d0=d['d0']
model_outputs = d['model_outputs']
raw_pred = model_outputs[1]
score = model_outputs[0]

img=viz_target(it.raw_img, it_anchor, d0.label[1].asnumpy(),model_outputs[1].asnumpy(),\
                         model_outputs[0].asnumpy(),d0.data[1].asnumpy(),show_num=None)

#ret_keep = mx_nms(raw_pred, it_anchor, score)









