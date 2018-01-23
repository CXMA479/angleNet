"""
this file is the further development of the training scrip (net-test.py)
in this code block, a fully training process is about to implement as well as
    visualization of the performance

i tried to take on the full resolution provided by git, as a result,
Author and Date is no longer needed.
"""

import mnxet as mx
import numpy as np
from config import config as cfg
cfg.it.debug.debug=  True



import sys, time, logging
from angleIter import angleIter
from make_symbol import gen_symbol, gen_model, feval_l1_angleMetric
from tool import mytick,viz_score_predict, viz_bbox_gdt, viz_target
import matplotlib.pyplot as plt


symbol_mod, fixed_param_names = gen_symbol()
it=angleIter(symbol_mod)#'../../data/angleList.txt','../../data/img',1,100)

mA_acc=mx.metric.create('acc') #mx.metric.CustomMetric(angleMetric)
mA_l1 = mx.metric.CustomMetric(feval_l1_angleMetric, name='l1_smooth loss')
mod = gen_model(symbol_mod,fixed_param_names,it)

t0=time.time()
for epoch_i in xrange(cfg.it.epoch):
  it.reset()
  for batch_i, d in enumerate(it):
    mod.forward_backward(d0)
    mod.update()
    out = mod.get_outputs()
    logging.info(out[0].asnumpy() )
    mA_acc.update([out[0]], [ d0.label[0]  ])
    mA_l1.update([out[1]], [ d0.label[1]  ])

    if batch_i % cfg.train.batch_size ==0 and batch_i is not 0: # update
      continue

    if batch_i % cfg.train.callbackBatch ==0 and batch_i is not 0: # log info
      t1=time.time()
      elapse = t1-t0
      logging.info('epoch[%d], batch:%d, acc:%f, rpn loss:%f, %.2f samples/sec'%\
          (epoch_i, batch_i,mA_acc.get()[1], mA_l1.get()[1], cfg.train.batch_size*cfg.train.callbackBatch*1./elapse ) )
      mA_acc.reset()
      mA_l1.reset()
      t0 = time.time()
      
  mod.save_checkpoint(cfg.train.save_prefix, epoch_i)
