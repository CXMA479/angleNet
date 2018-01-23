import matplotlib.pyplot as plt
import numpy as np
import mxnet as mx

img_path =  '../0.jpg'

img=mx.img.imdecode(  open(img_path,'rb'). read() ).asnumpy()

plt.imshow(img)
ax = plt. gca()
ax.text(500,400,'test', bbox=dict(facecolor='yellow', alpha=.7) )# fontsize=12
plt.show()















