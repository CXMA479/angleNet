"""
glue them together

Chen Y. Liang
Sep 6, 2017
"""
import mxnet as mx
import numpy as np
from config import config as cfg
import sys, logging, time
sys.path.append('it')
from angleIter import angleIter
from make_symbol import gen_symbol, gen_model
from tool import GroupMetrix

cfg.it.debug.debug=  not True

symbol_mod, fixed_param_names = gen_symbol()


logging.info(symbol_mod.tojson())


it=angleIter(symbol_mod)#'../../data/angleList.txt','../../data/img',1,100)
d=it.next()
#it.cur_it -= 1




mod = gen_model(symbol_mod,fixed_param_names,it)
GM  = GroupMetrix()
start_time =time.time()
for epoch_i in xrange(cfg.it.epoch):
  it.reset()
  for batch_id, d in enumerate(it):
    mod.forward(d)
    mod.acc_backward()
    GM.update(mod, d)
    if batch_id % cfg.train.batch_size == 0  and not (batch_id == 0) :
      mod.acc_update()

    if batch_id % cfg.train.callbackBatch==0 and batch_id >= cfg. train. callbackBatch:
      end_time = time.time()
      elapsed = end_time - start_time
      logging.info('Epoch[%d] Batch[%d]\tSpeed: %.2f samples/sec\tTrain-info: %s'%(\
                            epoch_i, batch_id, 1* cfg. train. callbackBatch/elapsed, GM.getMetric()))
      start_time  = time.time()
      GM.reset()

  if cfg.train.is_save:
    mod.save_checkpoint(cfg.train.save_prefix, epoch_i)

