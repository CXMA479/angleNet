import mxnet as mx
import numpy as np

x=mx.sym.Variable('x')
l=mx.sym.Variable('label')


data_x = np.random.randint(0,255,(80,120))
data_l = np.random.randint(-1,1,(80,))


y=mx.sym.FullyConnected(x, num_hidden=10)

symbol_mod = mx.sym.SoftmaxOutput(y,label=l, use_ignore=True, ignore_label=-1)

mod = mx.mod.Module(symbol,data_names=['x',], label_names=['label',])
mod.bind














