import mxnet as mx


sym = mx.sym.Variable('a')
sym = sym+1
sym = mx.sym.Convolution(sym,num_filter=2,kernel=(1,1),name='conv1')
sym = sym*2
sym = mx.sym.Convolution(sym,num_filter=2,kernel=(1,1),name='conv2')
sym = sym -10

sym_1 = sym.get_internals()['conv1_output']
sym_2 = sym.get_internals()['conv2_output']

sym= mx.sym.Group([sym_1, sym_2])

sym.list_arguments()
#['a', 'conv1_weight', 'conv1_bias', 'conv2_weight', 'conv2_bias']




