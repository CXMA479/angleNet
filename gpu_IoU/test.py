import mxnet as mx
import numpy as np


"""
x=mx.nd.reshape(mx.nd.array(xrange(10)), shape=(-1,1))
y=mx.nd.reshape(mx.nd.array(xrange(-5,0)),shape=(-1,1))
x_ones=mx.nd.ones((1,y.size))
y_ones=mx.nd.ones((1,x.size))

X=x*x_ones
Y=y*y_ones
Y=Y.T



>>> X

[[ 0.  0.  0.  0.  0.]
 [ 1.  1.  1.  1.  1.]
 [ 2.  2.  2.  2.  2.]
 [ 3.  3.  3.  3.  3.]
 [ 4.  4.  4.  4.  4.]
 [ 5.  5.  5.  5.  5.]
 [ 6.  6.  6.  6.  6.]
 [ 7.  7.  7.  7.  7.]
 [ 8.  8.  8.  8.  8.]
 [ 9.  9.  9.  9.  9.]]
<NDArray 10x5 @cpu(0)>

>>> Y

[[-5. -4. -3. -2. -1.]
 [-5. -4. -3. -2. -1.]
 [-5. -4. -3. -2. -1.]
 [-5. -4. -3. -2. -1.]
 [-5. -4. -3. -2. -1.]
 [-5. -4. -3. -2. -1.]
 [-5. -4. -3. -2. -1.]
 [-5. -4. -3. -2. -1.]
 [-5. -4. -3. -2. -1.]
 [-5. -4. -3. -2. -1.]]
<NDArray 10x5 @cpu(0)>
"""

l=np.zeros((1,4))
ll=np.repeat(l,2,axis=1)
#array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])





x=mx.nd.array(np.random.randint(0,10,(1,4)))
alpha = mx.nd.array(np.array([[1],[2],[3]]) )

x_a = mx.nd.broadcast_mul(x,alpha)

"""
>>> x

[[ 1.  8.  2.  6.]]
<NDArray 1x4 @cpu(0)>
>>> alpha

[[ 1.]
 [ 2.]
 [ 3.]]
<NDArray 3x1 @cpu(0)>
>>> x_a = mx.nd.broadcast_mul(x,alpha)
>>> x_a

[[  1.   8.   2.   6.]
 [  2.  16.   4.  12.]
 [  3.  24.   6.  18.]]
<NDArray 3x4 @cpu(0)>
>>> x_a = mx.nd.broadcast_mul(alpha,x)
>>> x_a

[[  1.   8.   2.   6.]
 [  2.  16.   4.  12.]
 [  3.  24.   6.  18.]]
<NDArray 3x4 @cpu(0)>
"""




