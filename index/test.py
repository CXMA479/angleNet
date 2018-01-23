import numpy as np
import mxnet as mx

a= [
     [      [1 , 2 , 3, 4],    [5 , 6 , 7, 8]     ],
     [      [9 , 10, 11, 12],  [13 ,14 ,15, 16]   ],
     [      [17 , 18 , 19, 20],  [21 , 22 , 23, 24]]
   ]  # 3 x 2 x 4
a=np.array(a)
a=mx.nd.array(a)
flt_a = mx.nd.reshape(a,(1,-1,2))
a=flt_a.asnumpy()
indx = a[0,:,1]> 3


