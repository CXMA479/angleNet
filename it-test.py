import mxnet as mx
import numpy as np
from angleIter import angleIter
from config import config as cfg

from make_symbol import gen_symbol, gen_model
from tool import mytick
cfg.it.debug.debug=  True

symbol_mod, fixed_param_names = gen_symbol()




it=angleIter(symbol_mod)#'../../data/angleList.txt','../../data/img',1,100)
d=it.next()


