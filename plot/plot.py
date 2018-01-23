import mxnet as mx

symbol=mx.sym.load('symbol_model.json')
mx.viz.plot_network(symbol).view()

