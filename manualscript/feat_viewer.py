"""
this object aimes to visualize the feature map for the given image,
    softmax vector is also supported in which case, (possible) class names are listed (top-K)
      mult-resolution input, specific output layer supports are planned to implemented.

Chen Y. Liang
Sep 18, 2017
"""

import mxnet as mx
import numpy as np
import os, logging, cv2, sys
import matplotlib.pyplot as plt

from   module import MutableModule


class Viewer (object):
  """
      for vgg: mean= [123.68, 116.28, 103.53], std=1, 224 for softmax
  """
  def __init__ (self,model_prefix, epoch, ctx=mx.gpu(), data_names=['data',], label_names=['prob_label',], mean=[0, 0, 0], std=1):
    self.sym, self.arg_params, self.aux_params = mx.model.load_checkpoint(model_prefix, epoch)
    self.ctx = ctx
    self.data_names = data_names
    self.label_names = label_names
    self.mean = np.array(mean)
    self.std  = std

    self.mod = None
    self.cut_symbol = None
    self.cut_symbol_name = None
    self.imgname  = None
    self.raw_img = None
    self.resized_img = None
    self.forward_img = None

    # containers for output
    self.raw_out_     = None   # hold raw output
    self.raw_out     = None    # hold post mean output
    self.resized_out = None    #  resize mean to raw image, set it to None once the output is a vector!


  def load_layers(self, symbol_name, bind_size=(224,224)):
    self.cut_symbol_name = symbol_name
    self.cut_symbol = self.sym.get_internals()['%s_output'%symbol_name]
    self.mod  = MutableModule(self.cut_symbol, self.data_names, self.label_names, context = self.ctx)
    self.mod.bind([( self.data_names[0],(1,3, bind_size[0], bind_size[1]) ), ], for_training = False)
    self.mod.init_params(arg_params = self.arg_params, aux_params = self.aux_params, allow_missing = False)

  def _predict(self, H0, W0, re_size_HW):
    if re_size_HW is not None: # force to resize
      self.resized_img = mx.img.imresize(self.raw_img, re_size_HW[1], re_size_HW[0])
    else:
      self.resized_img = self.raw_img
    self.resized_img = self.resized_img.asnumpy()
    self.raw_img = self.raw_img .asnumpy()

    self.forward_img = mx.nd.array( ( self.resized_img - self.mean )/self.std )
    self.forward_img = mx.nd.transpose(self.forward_img, axes=(2,0,1) )
    self.forward_img = mx.nd.expand_dims( self.forward_img, axis=0 )

    d= mx.io.DataBatch([self.forward_img,],provide_data=[(self.data_names[0],self.forward_img.shape),])
    self.mod.forward(d)

    # keepdims for mx.img.imresize
    self.raw_out_ = self.mod.get_outputs()[0]
    if len(self.raw_out_.shape)==2:  # softmax output
      self.resized_out  = None # referenced by view()
      self.raw_out = self.raw_out_[0].asnumpy()   #  ( num class, )

    elif len(self.raw_out_.shape) == 4: # normal
    #  raw feature map
      self.raw_out = mx.nd.mean(self.raw_out_[0], axis=0, keepdims=True) .asnumpy()  # eliminate batch axis
      self.raw_out = ( self.raw_out )/( self.raw_out.max() + np.exp(-8) )* 255
      self.raw_out = self.raw_out.astype(np.uint8) # 1 x h x w
    # resize...
      img_tmp = mx.nd.array(self.raw_out)
      img_tmp = mx.nd.transpose(img_tmp, axes=(1,2,0) )  # h x w x 1
      # resize to the original shape
      self.resized_out = mx.nd.transpose( mx.img.imresize(img_tmp, W0, H0), axes=(2,0,1)  )[0].asnumpy()  # h x w
      self.raw_out  = self.raw_out[0][:]    # h x w
    else:
    # Oop!
      assert 0


  def predict(self, img_path, re_size_HW=None):
    """ will not plot, use .view after this call"""

    assert os.path.isfile(img_path), '%s does not exist!'%img_path
    self.imgname = os.path.basename(img_path)
    self.raw_img = mx.img.imdecode( open(img_path, 'rb').read() )
    H0,W0 = self.raw_img.shape[:2]
    self._predict(H0, W0, re_size_HW)


  def view(self, block=False,top_k=5):
    assert self.raw_img is not None
    plt.figure()
    plt.suptitle(self.imgname)
    plt.subplot(221)
    plt.imshow(self.raw_img)
    plt.title('raw image')
    plt.subplot(223)
    plt.imshow(self.resized_img)
    plt.title('resized input')

    if self.resized_out is not None: # 1.raw 3.resized img 2. resized out 4. raw out
      plt.subplot(224)
      plt.imshow(self.raw_out)
      plt.title('raw feature map')
      plt.subplot(222)
      plt.imshow(self.resized_out)
      plt.title('resized feature map')
    else: # stem the vector
      plt.subplot(222)
      t=np.array(xrange(len(self.raw_out)))
      plt.stem(t, self.raw_out, marker='o')
      plt.title('softmax distribution')
      plt.subplot(224)
      idx = np.argsort(self.raw_out)[::-1]
      idx = idx[:top_k]
      t = t[idx]
      l = self.raw_out[idx]
      plt.stem(t,l,marker='o')
      label2show = 'most label: %s'%t
      plt.xlabel( 'most label: %s'%t)
      plt.title('Top-k distribution')
    plt.show(block=block)

  def crop_predict(self,img_path, xy_tl, xy_br, re_size_HW=None):
    """ will not plot, use .view after this call"""
    hw = (xy_br[1]-xy_tl[1], xy_br[0]-xy_tl[0])
    xy = xy_tl
    assert os.path.isfile(img_path), '%s does not exist!'%img_path
    self.imgname = os.path.basename(img_path)
    self.raw_img = mx.img.imdecode( open(img_path, 'rb').read() )
    self.raw_img = mx.img.fixed_crop(self.raw_img, xy[0], xy[1], hw[1], hw[0])
    H0,W0 = self.raw_img.shape[:2]
    self._predict(H0, W0, re_size_HW)
  def save_rawfeat(self, savename):
      plt.close()
      plt.matshow(self.raw_out)
      #plt.show()
      plt.savefig(savename)

if __name__ == '__main__':
  from viewer import Viewer
  import mxnet as mx
  import matplotlib.pyplot as plt
  v= Viewer('vgg16',0, mean=[123.68, 116.28, 103.53])
  v.load_layers('prob',bind_size=(224,224))
  v.predict('1.jpg',re_size_HW=(224, 224))
  v.view() 
  s=raw_input('press any key to exit')
  """
  v.crop_predict('1.jpg',(1114,18), (2457, 862 ),re_size_HW=(224, 224)) 
  v.view() 
  
  v= Viewer('vgg16',0)
  v.load_layers('relu5_3',bind_size=(224,224))
  v.predict('1.jpg', re_size_HW=(800,800)) 
  v.view() 
  
  v.predict('1.jpg',re_size_HW=(800,800)) 
  v.view() 
  """



