import mxnet as mx
import numpy as np

a=np.array([[1,2,3],[-3,-2,-1]])
b=np.array([[1,2,3],[-3,-2,-1]])*-10


l0= [mx.nd.array(a),mx.nd.array(b)]



l=[]
l.append(l0[0])
l.append(l0[1])


l0[0][0]=-100
l[0].asnumpy()
#array([[-100., -100., -100.],
#       [  -3.,   -2.,   -1.]], dtype=float32)

del l0

l=[mx.nd.array(a)*-2, -1*mx.nd.array(b)]
l[0].asnumpy()
