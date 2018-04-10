"""
      loads the trained model, predicts on the given image(s)
          filters the bbox w.r.t. the confidence
             view them on the figure

Chen Y. Liang
Sep 10, 2017
"""
import mxnet as mx
from module.module import MutableModule
import os, sys, tool
import numpy as np
from config import config as cfg
from angleIter import angleIter
import matplotlib.pyplot as plt
import logging
from nms.nms import mx_nms

class Viewer(object):
  """
         for angleNet prediction,
              support single image prediction and visualization
  """
  def __init__(self,model_prefix, epoch, pred_bbox_name='rpn_pred_bbox_reshape',\
                     pred_scor_name='rpn_label_loss', data_names=['data',],label_names=['target_label',]):

#model_prefix = '../output/'
#epoch=0
#
#pred_bbox_name = 'rpn_pred_bbox_reshape'
#pred_scor_name = 'rpn_label_loss'   # use output of the softmax
#data_names = ['data',]

    symbol_path ='%s-symbol.json'%model_prefix
    params_path ='%s-%04d.params'%(model_prefix,epoch)
    for x in [symbol_path, params_path]:
      assert os.path.isfile(x),'%s does not exist!'%x

    symbol = mx.sym.load(symbol_path)
    feature_symbol = symbol.get_internals()['%s_output'%cfg.net.rpn_conv_name]
    self.feature_symbol = feature_symbol
    fixed_param_names = feature_symbol.list_arguments()
    
    pred_bbox_sym = symbol.get_internals()['%s_output'%pred_bbox_name]
    pred_scor_sym = symbol.get_internals()['%s_output'%pred_scor_name]
    
    symbol = mx.sym.Group([pred_bbox_sym, pred_scor_sym])
#    print(symbol.list_arguments())
#    assert 0
    
    
#    it = angleIter(symbol,is_train=False)
    
    mod = MutableModule(symbol,data_names,label_names=label_names, context=cfg.predict.ctx)
    mod.bind([('data',(1,3,800,800)),], label_shapes =[], for_training=False)
    
#    _, arg_params, aux_params = mx.model.load_checkpoint(cfg.net.symbol_path,cfg.net.params_epoch)
    mod._curr_module.load_params(params_path)
    mod.params_initialized = True

    self.mod = mod

    self.model_prefix = model_prefix
    self.angleD=None
    self.HoW   =None
    self.sideLength = None
    self.imgname = None
    self.img     = None
    self.anchor  = None
    self.feat_shape = None
    self.stride    = None
    self.im_info   = None
    self.predict_bbox = None
    self.predict_score = None
    self.view_img = None
    self.filter_transfered_bbox = None
    self.indx = None

    self.predict_transfered_bbox = None  # by using tool.

    self.m = np.array(cfg.it.mean)
    self.parase_prefix()


  def parase_prefix(self):
    """
       xxx_angleD=xxx;HoW=xxx;SL=xxxx.log
         get angleD, HoW, sideLength
             exec('self'+xxx)
    """
    _, s= self.model_prefix.split('_')
#    s   = s[:-len('.log')]
    s_angleD,s_HoW, s_sideLength = s.split(';')
    for x in [s_angleD, s_HoW, s_sideLength]:
      exec('self.'+x)

  def predict(self,img_path):
    os.path.isfile(img_path),'%s does not exist!'%img_path

    self.imgname = os.path.basename(img_path)
    self.img = mx.img.imdecode(open(img_path,'rb').read()).asnumpy()

    img = mx.nd.array(self.img - self.m, cfg.predict.ctx)

    img = mx.nd.transpose(img, axes=(2,0,1))
    img = mx.nd.expand_dims( img, axis=0)
    self.im_info = np.array(self.img.shape[:2])
    _,feat_shape, _ = self.feature_symbol.infer_shape(data=img.shape)
    feat_shape = feat_shape[0]
    self.feat_shape = feat_shape
    self.stride=(self.im_info[0]/self.feat_shape[-2],self.im_info[1]/self.feat_shape[-1])
    self.anchor  = tool.genAnchor(self.im_info,self.angleD, self.HoW, self.sideLength, self.feat_shape, self.stride)
    logging.debug(img.shape)
#    assert 0
    d = mx.io.DataBatch([img],provide_data=[ ('data',img.shape),], provide_label=None)
    self.mod.forward(d)
   
    #self.raw_predict_bbox, self.raw_predict_score = [mx.nd.reshape(x,(-3,0)) for x in  self.mod.get_outputs()[:2] ]
    self.predict_bbox, self.predict_score = [mx.nd.reshape(x,(-3,0)).asnumpy() for x in  self.mod.get_outputs()[:2] ] 
    
    self.predict_transfered_bbox  = tool.bbox_inv_transfer(self.anchor, self.predict_bbox)
    #  filter who obviously out of boundary...
    pick_idx =  self.predict_transfered_bbox[:,0] < self.im_info[1]
    self.predict_transfered_bbox = self.predict_transfered_bbox[pick_idx,:]
    self.predict_score           = self.predict_score[pick_idx,:]
    pick_idx =  self.predict_transfered_bbox[:,0] >0
    self.predict_transfered_bbox = self.predict_transfered_bbox[pick_idx,:]
    self.predict_score           = self.predict_score[pick_idx,:]
#    logging.debug('x:'+str(pick_idx.sum()))

    pick_idx = self.predict_transfered_bbox[:,1] < self.im_info[0]
    self.predict_transfered_bbox = self.predict_transfered_bbox[pick_idx,:]
    self.predict_score           = self.predict_score[pick_idx,:]
    pick_idx = self.predict_transfered_bbox[:,1] > 0
    self.predict_transfered_bbox = self.predict_transfered_bbox[pick_idx,:]
    self.predict_score           = self.predict_score[pick_idx,:]
#    logging.debug('y:'+str(pick_idx.sum()))

    pick_idx =self.predict_transfered_bbox[:,3] < max(self.im_info)
    self.predict_transfered_bbox = self.predict_transfered_bbox[pick_idx,:]
    self.predict_score           = self.predict_score[pick_idx,:]
    pick_idx =self.predict_transfered_bbox[:,3]> 0 
    self.predict_transfered_bbox = self.predict_transfered_bbox[pick_idx,:]
    self.predict_score           = self.predict_score[pick_idx,:]
#    logging.debug('rh:'+str(pick_idx.sum()))
    

    pick_idx =self.predict_transfered_bbox[:,4] < max(self.im_info)
    self.predict_transfered_bbox = self.predict_transfered_bbox[pick_idx,:]
    self.predict_score           = self.predict_score[pick_idx,:]
    pick_idx =self.predict_transfered_bbox[:,4] > 0
    self.predict_transfered_bbox = self.predict_transfered_bbox[pick_idx,:]
    self.predict_score           = self.predict_score[pick_idx,:]
#    logging.debug('rw:'+str(pick_idx.sum()))
    pick_idx =self.predict_transfered_bbox[:,2] < np.pi
    self.predict_transfered_bbox = self.predict_transfered_bbox[pick_idx,:]
    self.predict_score           = self.predict_score[pick_idx,:]
    pick_idx =self.predict_transfered_bbox[:,2] > -np.pi
    self.predict_transfered_bbox = self.predict_transfered_bbox[pick_idx,:]
    self.predict_score           = self.predict_score[pick_idx,:]

  def view(self,filter_th,iou_th,block=False):
    #assert 0 < filter_th < 1
    #print self.raw_predict_bbox.shape, self.anchor.shape
    #assert 0
    self.indx = mx_nms(self.predict_bbox, None, self.predict_score,\
             self.predict_transfered_bbox, iou_th,score_thresh=filter_th)#, min_area=0, max_area=np.inf)
    
    # predict_score.shape :  1 x num x 2

    #self.indx = self.predict_score[:,1] > filter_th
    print('num of indexed bbox: %d'%len(self.indx))
    self.filter_transfered_bbox = self.predict_transfered_bbox[self.indx,:]
    img = self.img
    
    # nms...

    self.view_img = tool.draw_angleBox(img, self.filter_transfered_bbox.astype(np.float),(0,255,0))
#    plt.figure()
    plt.imshow(self.view_img)
    plt.title(self.imgname)
    plt.show(block=block)












