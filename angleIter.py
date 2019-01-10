import matplotlib
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import mxnet as mx
import os,sys
import numpy as np
import logging
import time
from tool import viz_bbox_gdt
import cv2

#sys.path.insert(0,'')   # tool.py
sys.path.insert(0,'cython')  # cytool.so
import tool
#import cytool

from config import config #logger
cfg = config
"""
try:
  os.system('make -C cython') # newest one
  from cytool import it_IoU  # cython implementation
except:
  os.system('rm cython/cytool.so')
  os.system('make -C cython')
  from cytool import it_IoU
"""
from gpu_IoU.gpu_IoU import gpu_it_IoU
nd= mx.nd
npr=np.random

def post_iou(iou_matrix,anchor,gdt):
  """
    iou_matrix: n x m

    anchor    : n x 5        x,y,alphaR,rh,rw
    gdt       : m x 6  label,

                   U must return the final version:

      return target_label, target_bbx, target_weight
  """
  anchor_num = anchor.shape[0]

  target_bbx    = anchor.copy()

  target_label  = np.empty( (anchor_num,) ,dtype=np.float32)

  #  1: objective, 0: background, -1: not care
  target_label.fill(-1)

  rpn_inside_weight  = np.zeros( (anchor_num,5),dtype=np.float32)
  rpn_outside_weight = np.zeros( (anchor_num,5),dtype=np.float32)
  if gdt.size > 0:

    # 1st. IoU>THRESHOLD_HIGH  ,label
    pred_idx, gdt_idx=np.where(iou_matrix>config.train.THRESHOLD_OBJECT)
#    logging.debug('[size] target_bbx gdt:'+str(target_bbx.shape)+' '+str(gdt.shape) )
    target_bbx[pred_idx, :] = gdt[gdt_idx, 1:]
#    target_weight[pred_idx][:] =1
    target_label[pred_idx]     =1     # only need to state wheather it is object or background

    # 2nd. remove IoU < THRESHOLD_LOW ,  label
    pred_idx=np.where(iou_matrix.max(1)<config.train.THRESHOLD_BACKGD)[0]
#    target_weight[pred_indx][:] =0
    target_label[pred_idx]     =0

    # 3rd. each gdt must have its own agent , label
    pred_idx=iou_matrix.argmax(0)
#    logging.debug('each gdt box\'s anchor idx:'+str(pred_idx))
    target_bbx[pred_idx,:] = gdt[:,1:]
#    target_weight[pred_indx][:] =1
    target_label[pred_idx]     =1
#    print(target_label.shape)
#    print(len(pred_idx))
#    fg_indx = np.where(target_label==1)[0]  # how many objectives are there
#    print(len(fg_indx))
#    assert 0


    # resolve conflict mappings, several anchors are allowed to share one gdt, but reversal is not allowed
    pred_idx_uniq,idx_cnts = np.unique(pred_idx,return_counts=True)
#    logging.debug('idx_cnt: %s, idx_uniq: %s'%(str(idx_cnts),str(pred_idx_uniq))  )
    conf_idx = np.where(idx_cnts>1)[0]
    if len(conf_idx)>0:
#      logging.debug('conflict idx:'+str(conf_idx))
      for ppidx in conf_idx: # recursive pointer, to keep thing simple, choose the maximum gdt
        anchor_idx = pred_idx_uniq[ppidx]
#        logging.debug('conflict entry: '+str(iou_matrix[anchor_idx]))
        gdt_max_idx = iou_matrix[anchor_idx,:].argmax()
#        logging.debug('gdt_max_idx:%d'%gdt_max_idx)
        target_bbx[anchor_idx,:]=gdt[gdt_max_idx,1:]

  else: # no target
    target_label[:]=0

  #   subsample...  keeping balance is important
  max_fg  = int(config.train.rpn_fg_frac * config.train.rpn_batch_size)
  fg_indx = np.where(target_label==1)[0]  # how many objectives are there
#  print(len(fg_indx))
#  logging.debug('ojectives num:%d'%len(fg_indx))
  logging.debug('objectives(%d), downsample...'%len(fg_indx))
  if len(fg_indx) > max_fg:
    #logging.debug('too many objectives(%d), downsample...'%len(fg_indx))
    disable_indx = npr.choice(fg_indx, len(fg_indx)-max_fg, replace=False)
    target_label[disable_indx] = -1

  # same story for backgd
  # use dynamic allocation...
  max_bg = int( len(fg_indx)*(1-config.train.rpn_fg_frac)/(config.train.rpn_fg_frac+1E-3)  )# config.train.rpn_batch_size - np.sum(target_label == 1)
  
#  logging.debug( ( len(fg_indx), max_bg ))
  bg_indx = np.where(target_label == 0)[0]
#  print(len(bg_indx))
#  assert 0


#  logging.debug('background num:%d'%len(bg_indx))
#  logging.debug('background (%d), downsample...'%len(bg_indx))
  if len(bg_indx) > max_bg:
    #logging.debug('too many background (%d), downsample...'%len(bg_indx))
    disable_indx = npr.choice( bg_indx, size=(len(bg_indx) - max_bg),replace=False)
    target_label[disable_indx] = -1

  rpn_inside_weight[target_label ==1, :]= np.array(config.train.rpn_bbx_weight)
  num_examples = np.sum(target_label>0)
  if not cfg.train.rpn_postive_weight >0: # use default (uniform)
    postive_weight  = np.ones( (1,5) )*1./num_examples
    negative_weight = np.ones( (1,5) )*1./num_examples
  else:
    assert 0<cfg.train.rpn_postive_weight <1
    postive_weight  = cfg.train.rpn_postive_weight/num_examples
    negative_weight = (1-cfg.train.rpn_postive_weight) /num_examples

  rpn_outside_weight[target_label==1,:] = postive_weight
  rpn_outside_weight[target_label==0,:] = negative_weight

  #  now nonlinear Transfer...
  if gdt.size > 0:
    target_bbx = tool.bbx_transfer(anchor,target_bbx)
#  logging.debug( ( sum(target_label>0), sum(target_label==0) ))
  return target_label, target_bbx, rpn_inside_weight, rpn_outside_weight



class angleIter(mx.io.DataIter):
  """
              labelFile
      file format:
      name.type;  l  x,y  alpha  rh  rw;  l...
        imgPath

        epoch

        it

        dataName
      'img','label','gdt'

  """
  def __init__(self,symGp=None,featName=None, is_train=True, labelFile=config.it.labelFile,\
            imgPath = config.it.imgPath):
    """
                        labelName: used for anchor-wise regression
      symGp : mx.symbol.Group([])
      stride: pixels(in raw img) between to neighbor anchors' centrs
      sideLength: \sqrt{Aera} , 
                                Aera is fixed for one sideLength wrt. angleD

      featName:  id of the feature map, used to identify the .npz file
    """

    if symGp is None:
      assert config.debug.debug

#    self.threshold_high=config.it.train.THRESHOLD_HIGH
#    self.threshold_low =config.it.train.THRESHOLD_LOW

    super(angleIter,self).__init__()
    self.x_num  =config.train.iou_x_num
    self.angleD = config.ANCHOR_angleD
#    self.stride = config.ANCHOR_stride
    self.HoW    = config.ANCHOR_HoW
    self.sideLength = config.ANCHOR_sideLength
    self.epoch=config.it.epoch
    self.it=config.it.iteration
    self.shuffle=config.it.shuffle
    self.is_train = is_train
    self.batch_size  = 1

    self.dataNames=config.it.dataNames
    self.labelNames=config.it.labelNames

    self.data_shapes  = []
    self.label_shapes = []

    assert os.path.isfile(labelFile),labelFile
    self.fileName = labelFile
    assert os.path.isdir(imgPath), imgPath
    self.imgPath = imgPath

    if config.it.debug.feat_shape is not None:
      assert cfg.it.debug.debug is True
      self.feat_shape = config.it.debug.feat_shape
      self.stride=None
    else:
      assert cfg.ANCHOR_stride is  None
      self.stride=cfg.ANCHOR_stride    # calculate on the fly

    if featName is None:
#      assert config.it.debug.debug
      self.featName = ''
    else:
      self.featName = featName

    self.fs=open(self.fileName).readlines()
    self.imgNum=len(self.fs)
#    assert self.imgNum == len(os.listdir(self.imgPath))
    self.cur_it=0
    self.cur_epoch=0
    self.lineIdx=0   #   line index of the img
    self.lineStr=''
    self.imgName=''
    self.lg=''
    self.provide_label=None#[( 'img']
    self.provide_data=None#[]


    # get the symobl ofthe feature map
    self.feat_sym=symGp.get_internals()[config.net.rpn_conv_name+'_output']  if config.it.debug.feat_shape is None else None
#    self.feat_sym=symGp if config.it.debug.feat_shape is None else None
      
    self.target_label=None
    self.target_gdtbox=None
#    self.target_weight=None
    self.rpn_inside_weight=None
    self.rpn_outside_weight=None
    self.gdt=None
    self.im_info=None
    self.img=None
    self.mean = np.array(cfg.it.mean)
    self.type_num = len(self.HoW)*len(self.angleD)*len(self.sideLength)
    iouprefix_name = str(config.ANCHOR_angleD)+str(config.ANCHOR_HoW) + str(config.ANCHOR_sideLength)+\
        str(config.ANCHOR_stride)

    self.ioudir = os.path.join(config.it.ioudir,'train',iouprefix_name) if is_train else os.path.join(config.it.ioudir,'test',iouprefix_name)
    self.secure_anchor_iou_file() # clean interface

    if config.it.debug.debug:
      self.anchor=0
    self.anchor=None
    self.iou_matrix=None

  def gen_iou(self):
    logging.info('generate iou files ...')
    
  def load_anchor_iou(self,imgName,featName):
    ioufile= self.gen_ioufileDir(imgName,featName)
    assert '.npz' in ioufile,'%s is not a .npz file.'%ioufile
    ld= np.load(ioufile)
    return ld['anchor'],ld['iou']

  def secure_anchor_iou_file(self):
    """
      integrated function to secure ious' supply
    """
    # check all the image files, if its .npy is not presented, gen it rightaway 
    logging.info('check if all the iou files are ready...')
    for self.lineIdx in xrange(self.imgNum):
      self.lineStr=self.fs[self.lineIdx]
      if ';' not in self.lineStr:
        continue
      self.lg = self.lineStr.rsplit(';\t')
      self.imgName = self.lg.pop(0)

      ioufile = self.gen_ioufileDir(self.imgName,self.featName)
      if '.npz' in ioufile: # already exists
        continue
      # if not, gen it...
      logging.info('.npz missed for %s, gen it...'%self.imgName)
      self.parseLine()
      if self.stride is None:
        self.stride=(self.im_info[0]/self.feat_shape[-2],self.im_info[1]/self.feat_shape[-1])

      anchor = tool.genAnchor(self.im_info,self.angleD, self.HoW, self.sideLength,\
              self.feat_shape,self.stride) # anchor : N x 5
#      iou_matrix = it_IoU(anchor,     self.gdt[:,1:],   \
#                self.x_num)    #  BE CAREFUL ! [:,0] IS FOR LABEL !

      iou_matrix = gpu_it_IoU(anchor,     self.gdt[:,1:],   \
                self.x_num)    #  BE CAREFUL ! [:,0] IS FOR LABEL !

               
      np.savez(ioufile,anchor=anchor,iou=iou_matrix)
      logging.info('saved at %s'%ioufile)
    logging.info('ok.')

  def gen_ioufileDir(self,imgName,featName):
    assert '.' in imgName
    ioufile=self.ioudir+'_'+imgName[:-4]+featName+'_anchor_iou.npz'# store anchor and iou
    if  not os.path.isfile(ioufile): # prepare to gen, no suffix of .npy
      return self.ioudir+'_'+imgName[:-4]+'_anchor_iou'   
    else:
      return ioufile

  def reset(self):
    self.cur_it=0


  def checkAndSet(self):
    if self.cur_it >= self.it:
      self.cur_epoch +=1
      raise StopIteration
    self.cur_it +=1

  def genIdx(self):
    if self.shuffle:
      self.lineIdx = np.random.randint(0,self.imgNum)
    else:
      self.lineIdx +=1
      if self.lineIdx >= self.imgNum:
        self.lineIdx =0


  def parseLine(self):
    """
                       all convert to matrix, including img (read it and convert to matrix)
               and get the feature map's shape
    """
    self.lineStr=self.fs[self.lineIdx]
    while ';' not in self.lineStr:
        self.lineIdx += 1
        self.lineStr = self.fs[self.lineIdx]
    self.lg = self.lineStr.rsplit(';\t')
    self.imgName = self.lg.pop(0)
    self.img=mx.image.imdecode(open(os.path.join(self.imgPath,self.imgName),'rb').read())
    self.img = mx.nd.array( self.img.asnumpy()-self.mean)


    self.lgSize=len(self.lg)   # as Batch Channel
    self.imgHWC=self.img.shape
#    if config.debug.debug:
#      logging.debug('image shape:'+str(self.imgHWC))
    H,W,C=self.imgHWC
    if self.feat_sym is not None:
      _, self.feat_shape, _=self.feat_sym.infer_shape(**{self.dataNames[0]:(1,C,H,W)})  # it only needs img
      self.feat_shape = self.feat_shape[0]

    self.provide_data=[(self.dataNames[0],(1,C,H,W)), \
           (self.dataNames[1],(self.lgSize,1,1,6)),\
           (self.dataNames[2],(2,))]    # H,W

    #     label, bbx, weight
    feat_shape = self.feat_shape
#    logging.debug('feat_shape:%s'%(str(feat_shape)))
#    self.provide_label=[(self.labelNames[0],(feat_shape[-2]*feat_shape[-1]*self.type_num,1)),\
#            (self.labelNames[1],(feat_shape[-2]*feat_shape[-1]*self.type_num,5))]#,\
#            (self.labelNames[2],(feat_shape[-2]*feat_shape[-1]*self.type_num,5))]


    self.gdt=np.zeros((self.lgSize,6))
    self.im_info=np.array([H,W])

    self.lg.append(self.lg.pop(-1).strip())  # get rid of '\n'
    #   go for 'label', 'gdt'
    for i,obj in enumerate(self.lg):
      """
                                 now obj:  label  x,y  alpha(deg)  rh  rw
      """
      label,xy,alphaD,rh,rw=obj.split('\t')
      label,alphaD,rh,rw = [ np.float(_) for _ in (label,alphaD,rh,rw)]
      x,y=[np.float(_) for _ in xy.rsplit(',') ]
      alpha=np.deg2rad(alphaD)  # deg 2 rad
      self.gdt[i][:] = np.array([label,x,y,alpha,rh,rw])


  def checkout_ioufile(self):
    return os.path.isdir(self.ioudir)


  def next(self):  # one img per call
    self.checkAndSet()
    self.genIdx()
    self.parseLine()
    #   havnt ended yet! longway to go
    #   assign gdt for each anchor, but 1st, u need generate anchors
    
    #   an_gdtbox IS NOT gdtbox FOR NOW !

    self.stride=(self.im_info[0]/self.feat_shape[-2],self.im_info[1]/self.feat_shape[-1])
    self.anchor, self.iou_matrix = self.load_anchor_iou(self.imgName,self.featName)

    # checksum...
    


#    if config.it.debug.debug:
#      self.anchor=anchor



#    if self.hasiou_flag and not os.path.isfile(ioufile): # need to notify the owner
#      logging.info('ioufile does not exsists: '+ioufile)

    img=self.img.asnumpy()+self.mean
    self.raw_img = img.astype(np.uint8)
    self.target_label,self.target_bbx, self.rpn_inside_weight, self.rpn_outside_weight =\
                                         post_iou(self.iou_matrix,self.anchor,self.gdt)
    """
    if config.it.debug.debug:
#      logging.debug('[IoU] time elapsed:'+str(time.time()-start))
#      logging.debug('[IoU] maximum:%f'%np.max(self.iou_matrix))
#      logging.debug('iou matrix:'+str(iou_matrix))
#      logging.debug('begin to calculate IoU...')
#      logging.debug('anchor:'+str(anchor))
#      logging.debug('gdt:'+str(self.gdt))

      #  display the img with anchors and gdt onit !
      show_anchor_indx = np.random.randint(0, self.anchor.shape[0], cfg.debug.it.anchor_num)
      
      img=tool.draw_angleBox(img.astype(np.uint8),self.anchor[show_anchor_indx][:],(0,250,0))
      img=tool.draw_angleBox(img,self.gdt[:,1:],(0,0,250))
      plt.imshow(img)
      ax=plt.gca()  # ax.text(x,y,text)

      iou_list = self.iou_matrix[show_anchor_indx][:].max(axis=1)

      xy_list  = self.anchor[show_anchor_indx,:2]
#      logging.debug('iou list shape: '+str(iou_list.shape))
#      logging.debug('xy list shape: '+str(xy_list.shape))
#      logging.debug('anchor shape: '+str(self.anchor.shape))
#      assert 0

      tool.text_fig(ax,xy_list, iou_list)
      plt.figure()
#      print(self.anchor.shape, np.expand_dims(self.target_bbx,axis=0).shape,\
#              np.expand_dims(self.rpn_outside_weight,axis=0).shape)
#      img = viz_bbox_gdt(self.raw_img, self.anchor, np.expand_dims(self.target_bbx,axis=0),\
#                np.expand_dims(self.rpn_outside_weight,axis=0), show_num=20)

#      plt.show()

    """
#    print(self.target_label.shape)
#    assert 0

    self.img=nd.transpose(self.img,axes=(2,0,1))


    self.img=nd.expand_dims(mx.nd.array(self.img,cfg.train.ctx),axis=0)
    self.rpn_inside_weight = nd.expand_dims(mx.nd.array(self.rpn_inside_weight,cfg.train.ctx),axis=0)
    self.rpn_outside_weight = nd.expand_dims(mx.nd.array(self.rpn_outside_weight,cfg.train.ctx),axis=0)

    self.target_label = nd.expand_dims(mx.nd.array(self.target_label,cfg.train.ctx),axis=0)
    self.target_bbx = nd.expand_dims(mx.nd.array(self.target_bbx,cfg.train.ctx) ,axis=0)

#    self.provide_label = [('img', self.img.shape ), ('gdt', self.gdt.shape), ('im_info',self.im_info.shape)]
#    self.provide_data = [('label',self.target_label.shape), ('bbx',self.target_bbx),('weight', self.target)weight.shape)]

#    self.provide_data  =[(self.dataNames[0], tuple(self.img.shape)),\
#                         (self.dataNames[1],tuple(self.gdt.shape) ),\
#                         (self.dataNames[2],tuple(self.im_info)),\
#                         (self.dataNames[3],tuple(self.rpn_inside_weight.shape)),\
#                         (self.dataNames[4],tuple(self.rpn_outside_weight.shape))]
#    self.provide_label =[(self.labelNames[0],tuple(self.target_label.shape)),\
#                         (self.labelNames[1],tuple(self.target_bbx.shape))]

    self.provide_data  =[(self.dataNames[0], tuple(self.img.shape)),\
                         (self.dataNames[1],tuple(self.rpn_inside_weight.shape)),\
                         (self.dataNames[2],tuple(self.rpn_outside_weight.shape))]
    self.provide_label =[(self.labelNames[0],tuple(self.target_label.shape)),\
                         (self.labelNames[1],tuple(self.target_bbx.shape))]
#    return mx.io.DataBatch( [self.img,self.gdt,self.im_info, self.rpn_inside_weight, self.rpn_outside_weight],[self.target_label,self.target_bbx ],provide_data=self.provide_data, provide_label=self.provide_label)
    return mx.io.DataBatch( [self.img, self.rpn_inside_weight, self.rpn_outside_weight],[self.target_label,self.target_bbx ],provide_data=self.provide_data, provide_label=self.provide_label)


