"""
       a mount of handles need to be created
                Chen Y. Liang
"""

import mxnet as mx
import numpy as np
import logging
import sys,os, time, cv2
import matplotlib.pyplot as plt
sys.path.insert(0,'cython')


try:
  from cytool import cp2p
except:
  os.system('rm cython/cytool.so')
  os.system('make -C cython')
  from cytool import cp2p
import cv2

def XXX_genAnchor(im_info,angleD,HoW,sideLength,feat_shape,stride):
  assert 0, 'Deprecated. Use genAnchor'
  """
                keep all anchors' centrs inside the RAW img

    return  anchor  #  N x 5

  """
  assert im_info.size==2 # to secure consistency: H,W
  assert len(stride)==2   # stride_h , stride_w
  feat_H,feat_W = feat_shape[-2:]
#  logging.debug('img_info:'+str(im_info))
  raw_H,raw_W   = im_info[-2:]
  angle=np.deg2rad(angleD)

  type_num   =len(angleD)*len(HoW)*len(sideLength)
  anchor_num = feat_H*feat_W*type_num

#  anchors = np.zeros( (1,anchor_num,1,5) )  #  x,y,angle,rh,rw

#  index=np.array(0,anchor_num,type_num)

  x = np.array(xrange(feat_W))*.0+1#*stride[1]
  y = np.array(xrange(feat_H))*.0+1#*stride[0]

  x[0]  = 0.5#stride[1]/2
#  x[-1] -= stride[1]/2

  y[0]  =0.5# stride[0]/2
#  y[-1] -= stride[0]/2


#  logging.debug(x)
  x=np.round(x.cumsum()*stride[1])
  y=np.round(y.cumsum()*stride[0])


#  logging.debug(stride)
#  logging.debug(x)

  X,Y = np.meshgrid(x,y)

  # filter
  x[x>=raw_W] = raw_W-1
  y[y>=raw_H] = raw_H-1

  an_X = np.repeat( X.flatten()[np.newaxis,:], type_num, axis=1).flatten()
  an_Y = np.repeat( Y.flatten()[np.newaxis,:], type_num, axis=1).flatten()

  an_angle = np.repeat( np.array(angle)[np.newaxis,:],feat_H*feat_W, axis=0).flatten()
#  logging.debug('an_angle:'+str(an_angle))

  an_angle = np.repeat( an_angle,type_num/len(angle), axis=0)

  s=np.array(sideLength)
  a=np.array(HoW)
  """
    h/w=a    (HoW)
    hw=s**2  (area)

    ===>  w= h/a = s**2/h
      ===>  h**2 = a * s**2
        ===>   h= s* sqrt(a)
          ===>   w = s/sqrt(a)
  """
  an_rh   =  np.reshape(s,(-1,1)) * np.reshape(   np.sqrt(a)  ,(1,-1))/2
  an_rw   =  np.reshape(s,(-1,1)) / np.reshape(   np.sqrt(a)  ,(1,-1))/2

#  logging.debug('an_rh:'+str(an_rh))
#  logging.debug('an_rw:'+str(an_rw))
  an_rh=np.repeat( an_rh.flatten()[np.newaxis,:], len(angle)* feat_H*feat_W,axis=0 ).flatten()

#  logging.debug('an_rh:'+str(an_rh))

#  an_rh=np.repeat( an_rh, feat_H*feat_W,axis=0 )  

  an_rw=np.repeat( an_rw.flatten()[np.newaxis,:], len(angle)*feat_H*feat_W,axis=0 ).flatten()
#  logging.debug('an_rw:'+str(an_rw))


#  an_rw=np.repeat( an_rw, feat_H*feat_W,axis=0 )

  anchor= np.vstack( (an_X, an_Y, an_angle, an_rh, an_rw))  #  (n,) strenches towards horizon!
#  logging.debug('anchor:'+str(anchor.transpose()))
#  logging.debug('anchor shape:\n'+str(anchor.transpose().shape))
  anchor= anchor.transpose()

  # re-index by map-wise --   prepare for the prediction layers
  map_idx=[]
  for type_idx in xrange(type_num):
    map_idx = map_idx + list(xrange(type_idx,anchor_num,type_num ))


  return anchor[map_idx]

def genAnchor(im_info,angleD,HoW,sideLength,feat_shape,stride):
    type_num = len(HoW)*len(sideLength)*len(angleD)
    feat_H,feat_W = feat_shape[-2:]
    x = np.arange(feat_W)*.0+1
    y = np.arange(feat_H)*.0+1

    x[0]  = 0.5#stride[1]/2
    y[0]  =0.5# stride[0]/2
    x=np.round(x.cumsum()*stride[1])
    y=np.round(y.cumsum()*stride[0])


    X,Y = np.meshgrid(x,y)  # (28,28)
#    print X,Y
    X,Y = [np.repeat( np.expand_dims(Z, axis=0),repeats=type_num,axis=0) for Z in [X,Y] ]

    X,Y = [np.transpose(Z, axes=(1,2,0) ).reshape((-1,)) for Z in [X,Y]  ]
    angle=np.deg2rad(angleD)
    an_angle = np.array(angle)
    s=np.array(sideLength)
    a=np.array(HoW)
    an_rh  =  (np.reshape(s,(-1,1)) * np.reshape(   np.sqrt(a)  ,(1,-1))/2).reshape((-1,))
    an_rw  =  (np.reshape(s,(-1,1)) / np.reshape(   np.sqrt(a)  ,(1,-1))/2).reshape((-1,))
#    print an_rh.shape
    an_rh,_ = np.meshgrid( an_rh, an_angle )
    an_rw, an_angle = np.meshgrid(an_rw, an_angle)
    anchor = np.vstack(  (an_angle.flatten(), an_rh.flatten(), an_rw.flatten())  ).T
#    print anchor#.shape
    pixel_num = X.shape[0]/type_num
    anchor = np.tile(  anchor, (pixel_num,1))
#    print anchor[:20]#.shape
    anchor_1 = np.vstack((X,Y)).T
#    print anchor.shape, anchor_1.shape, pixel_num, type_num, an_angle.shape
#    assert 0
    return np.concatenate( (anchor_1,anchor), axis=1  )

def  bbx_transfer(anchor,anchor_gdt):
  """
                  anchor     : n x 5
      anchor_gdt : n x 5   # data copyed from gdt if objective(diff from anchor)

      t_x *  :  (x-x_a)/ ( rh_a  )
      t_y *  :  (y-y_a)/ ( rh_a  )
      -----alpha *:  ln( ( alpha+ 2pi) / (alpha* + 2pi) )----

      alpha* :  alpha - alpha_a

      t_rh * :  ln( rh / rh_a )
      t_rw * :  ln( rw /rw_a )
  """
  target_bbx = np.empty((anchor.shape[0],5),dtype=np.float32)

  target_bbx[:,0] = (anchor_gdt[:,0] - anchor[:,0]) / (anchor[:,3]  )
  target_bbx[:,1] = (anchor_gdt[:,1] - anchor[:,1]) / (anchor[:,3]  )

#  target_bbx[:,2] = np.log( (anchor_gdt[:,2] + 2*np.pi) / (anchor[:,2] + 2*np.pi) )  # log -> ln
  target_bbx[:,2] = anchor_gdt[:,2] - anchor[:,2]

  target_bbx[:,3] = np.log( (anchor_gdt[:,3] +np.exp(-6) )/( anchor[:,3] +np.exp(-6)) )
  target_bbx[:,4] = np.log( (anchor_gdt[:,4] +np.exp(-6) )/( anchor[:,4] +np.exp(-6)) )

  return target_bbx

def bbox_inv_transfer(anchor,predict_bbx, xpu='cpu'):
    """
              inverse transfermation for prediction (for view)

         view_bbox -> anchor_gdt,  predicr_bbx  -> target_bbx
    """
    view_bbox = np.empty((anchor.shape[0],5),dtype=np.float32) if xpu is 'cpu' else \
                mx.nd.empty((anchor.shape[0],5),dtype=np.float32)

    view_bbox[:,0] = predict_bbx[:,0] * anchor[:,3] + anchor[:,0]
    view_bbox[:,1] = predict_bbx[:,1] * anchor[:,3] + anchor[:,1]

  #  view_bbox[:,2] = np.exp(predict_bbx[:,2])*(anchor[:,2]+ 2*np.pi) - 2*np.pi
    view_bbox[:,2] = predict_bbx [:,2] + anchor [:,2]
    exp = np.exp if xpu is 'cpu' else mx.nd.exp

    view_bbox[:,3] = exp(predict_bbx[:,3])*anchor[:,3]
    view_bbox[:,4] = exp(predict_bbx[:,4])*anchor[:,4]
    return view_bbox

def draw_angleBox(img,anchor,color, line_width=1):
  """
                img   : H x W x 3 , RGB
                anchor: N x 5     , x,y,alphaR,rh,rw
                color: [r,g,b]
  """

  retImg=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

  num=anchor.shape[0]
  for i in np.arange(num):
#    print anchor[i][:2].shape
#    assert 0
    P = cp2p(anchor[i][:2],anchor[i,2],anchor[i,3],anchor[i,4])  #   2 x 4
    P = np.round(P).astype(np.int)
    for i in np.arange(3):
      cv2.line(retImg,tuple(P[:,i]),tuple(P[:,i+1]),color,line_width)
    cv2.line(retImg,tuple(P[:,3]),tuple(P[:,0]),color,1)

  return cv2.cvtColor(retImg,cv2.COLOR_BGR2RGB)

def text_fig(ax, xy_list, iou_list):
  for xy, iou in zip(xy_list, iou_list):
    ax.text(xy[0], xy[1], '%.2f'%iou, bbox=dict(facecolor='yellow',alpha=.6) )

def viz_target(img, anchors, anchor_gdt, predict_bbox, scores, rpn_inside_weight,  show_num=None):
    """
        display the bbox whose weight is postive for debuging
            anchors are also shown
        img:
            (r,g,b)
        anchors:
            (hw x num, 5)
        anchor_gdt:
            (1, hw x num, 5)
        predict_bbox:
            (1, hw x num, 5)
        rpn_inside_weight:
            (1,hw x num, 5)
        scores:
            (1, hw x num, 2)
    """
    valid_idx = np.where(rpn_inside_weight[0,:,0]>0)[0]
    if show_num is not None:
        np.random.shuffle(valid_idx)
        valid_idx = valid_idx[:min(show_num, len(valid_idx))]
    from nms.nms import mx_nms
#    print('%d boxes t')
    
    sub_valid_idx = mx_nms(mx.nd.array(predict_bbox[:,valid_idx,:]),anchors[valid_idx], mx.nd.array(scores[:,valid_idx,:]))
    print('%d boxes lefted.'%len(valid_idx))
    valid_idx = valid_idx[sub_valid_idx]
#    valid_idx = valid_idx[0]
#    print(valid_idx, valid_idx.shape, rpn_outside_weight.shape)
#    obj_idx = np.where(rpn_inside_weight[0,valid_idx,0]>0)
    pred_box = bbox_inv_transfer(anchors[valid_idx,:],predict_bbox[0][valid_idx,:])
    gdt_box = bbox_inv_transfer(anchors[valid_idx,:], anchor_gdt[0][valid_idx,:])
#    anc_box = bbox_inv_transfer(anchors[valid_idx,:], anchors[valid_idx,:])
    # object gdt is shown as red
    img=draw_angleBox(img, gdt_box.astype(np.float), [255,0,0]) # cv2 use bgr...  -_-||
    img=draw_angleBox(img, pred_box.astype(np.float), [0,0,255])
    img= draw_angleBox(img, anchors[valid_idx,:],[112, 255, 125]) # see org anchors for referencing...
    plt.imshow(img)
#    text_fig(plt.gca(), pred_box[:,:2],scores[0,valid_idx,1])
    plt.title('object: R, GD-T: B, anchor: G, num: %d'%gdt_box.shape[0])
    plt.show(block=False)
    return img

def viz_anchor_byIoU(img, anchors, raw_gdt, IoU, th1, th2=1, title_s='', text=True):
    iou_idx = (np.max(IoU, axis=1)< th2) *( np.max(IoU, axis=1)> th1)
    picked_anchor = anchors[iou_idx,:]
    picked_iou = np.max(IoU[iou_idx,:],axis=1)
#    picked
#    gdt_box = bbox_inv_transfer(picked_anchor, picked_gdt)
#    print (raw_gdt[0], raw_gdt.shape)
#    print [iou_idx,:],iou_idx
#    assert 0
    gdt = bbox_inv_transfer(anchors[iou_idx,:],raw_gdt[iou_idx,:] )
    img =draw_angleBox(img, gdt.astype(np.float), [0, 0, 255])
    img =draw_angleBox(img, picked_anchor, [112, 255, 125] )
    plt.imshow(img)
    if text:
        text_fig(plt.gca(), picked_anchor[:,:2], picked_iou)
    title_s = title_s+'th:%f-%f,#anchor:%d, green: anchors, red: gdt'%(th1,th2, picked_iou.size)
    plt.title(title_s)
    #plt.show(False)

def mapgdt(iou,th1, th2=1,feat_hw=(28,28), anchor_num=6,block=False):
    iou_idx = (np.max(iou, axis=1)< th2) *( np.max(iou, axis=1)> th1)
    iou_idx = (iou_idx.reshape(feat_hw+(anchor_num,)).mean(axis=-1)*255).astype(np.uint8)
#    plt.figure()
    plt.imshow(iou_idx)
    plt.show(block=block)

def check_order(feat_hw=(28,28),stride=(16,16), anchor_num=6,dim=5):

    x = np.array(xrange(feat_hw[0]))*.0+1#*stride[1]
    y = np.array(xrange(feat_hw[1]))*.0+1#*stride[0]

    x[0]  = 0.5#stride[1]/2
    y[0]  =0.5# stride[0]/2
    x=np.round(x.cumsum()*stride[1])
    y=np.round(y.cumsum()*stride[0])


    X,Y = np.meshgrid(x,y)  # (28,28)
#    print X,Y
    X,Y = [np.expand_dims(Z, axis=0) for Z in [X,Y] ]
#    return None,None,X,Y
    for channel_idx in range(dim*anchor_num-1): #  (6#,28,28)
#       print X[0]*10+channel_idx
#        print np.sum(np.abs(X-Y))
        X,Y = [ np.concatenate( (Z,np.expand_dims(Z[0]*100+channel_idx, axis=0)) ) for Z in [X,Y] ]
#    print X.shape
    # (28,28,6#) -> (28x28x#,6)
#    return None,None,X,Y
    return X, Y
    X_re,Y_re = [np.transpose(Z, axes=(1,2,0) ).reshape((-1,anchor_num)) for Z in [X,Y]  ]
    print '*'*20
#    X_re,Y_re = [Z.reshape((-1,)) for Z in [X_re,Y_re]]
    print np.sum(np.abs(X_re-Y_re))
    return [X_re,Y_re]#,X,Y]

def viz_bbox_gdt(img, anchors, anchor_gdt, rpn_outside_weight,  show_num=None):
    """
        frankly, this one should be inside the angleIter,
            but it is found to be neccessary only in this stage
        check anchor_gdt where rpn_outside_weight is postive via plotting

        img:
            (r,g,b)
        anchors:
            (hw x num, 5)
        anchor_gdt:
            (1, hw x num, 5)
        rpn_outside_weight:
            (1,hw x num, 5)
        rpn_inside_weight:
            ~ ~
    """
    valid_idx = np.where(rpn_outside_weight[0,:,0]>0)[0]
    if show_num is not None:
        np.random.shuffle(valid_idx)
        valid_idx = valid_idx[:min(show_num, len(valid_idx))]
#    valid_idx = valid_idx[0]
#    print(valid_idx, valid_idx.shape, rpn_outside_weight.shape)
#    obj_idx = np.where(rpn_inside_weight[0,valid_idx,0]>0)
    gdt_box = bbox_inv_transfer(anchors[valid_idx,:], anchor_gdt[0][valid_idx,:])
    # object gdt is shown as red
    img = draw_angleBox(img, gdt_box.astype(np.float),[0, 0, 255])
    plt.imshow(img)
    plt.title('object: red, backGND: blue, num: %d'%gdt_box.shape[0])
    plt.show()




def viz_score_predict(img, anchors, anchor_gdt, predict_bbox, scores, score_th=.9, ax=plt.gca(), show_num=None ):
    """
        draw predicted bbox(es) whose scores are above threshould
            bbox_inv_transfer is invoked to restore the bbox from anchor and gdt/prediction
        all args are as type of numpy

        img:
            (H,W,[r,g,b]) np.uint8
                it is recommanded to keep img as a clean one for better visualization
        anchors:
            (hw x num, 5)
        anchor_gdt:
            (1, hw x num, 5)
        predict_bbox:
            (1, hw x num, 5)
        scores:
            (1, hw x num, 2)
    """
    if anchor_gdt is not None:
        assert scores.shape[2]==2 and anchor_gdt.shape[0]==1
    assert img.dtype==np.uint8
    valid_idx = scores[0]>score_th
    valid_idx = valid_idx[:,1]
    if show_num is not None:
#        print('check show num...')
        true_idxes = np.where(valid_idx)[0]
#        print(true_idxes,len(true_idxes),show_num)
        if len(true_idxes)>show_num:
            np.random.shuffle(true_idxes)
            valid_idx[:] = False
#            print('all set to 0',sum(valid_idx))
            true_idxes = true_idxes[:show_num]
            for idx in true_idxes:
#                print(idx)
                valid_idx[idx] = True
#            print('set some to true',sum(valid_idx))
#    print(anchors[valid_idx,:].shape, sum(valid_idx),valid_idx)
    if anchor_gdt is not None:
        gdt_box = bbox_inv_transfer(anchors[valid_idx,:], anchor_gdt[0][valid_idx,:])
        img=draw_angleBox(img, gdt_box.astype(np.float), [255,0,0]) # cv2 use bgr...  -_-||
        title_s = 'gdt: blue, pred: red, obj-scores'
    else:
        title_s = 'pred: red, obj-scores'
    pred_box = bbox_inv_transfer(anchors[valid_idx,:],predict_bbox[0][valid_idx,:])
    # draw on the image...
    ## gdt is blue, while predict is red...

    img=draw_angleBox(img, pred_box.astype(np.float), [0,0,255])
    ## attach scores...
    plt.imshow(img)
    text_fig(ax, pred_box[:,:2],scores[0,valid_idx,1])
    plt.title(title_s)
    plt.show()








class mytick(object):
  def __init__(self,proc='proc',logging=None):
    self.cur=0
    self.s_time =None# time.time()
    self.eplapse =0
    self.proc=proc
    self.logging=logging
  def load(self):
    self.s_time=time.time()
    logging.info('starting the process, please wait for the progress bar...')
    
  def tick(self,idx,num):
    cur_tmp = int(idx*100/num)
    if cur_tmp > self.cur:
      self.cur = cur_tmp
      self.elapse = round(time.time() - self.s_time)
      self.s_time = time.time()
      logging.info('%s:\t %d%s elapsed:\t%d sec'%(self.proc,self.cur,'% ,',int(self.elapse)))





#def it_IoU(anchor,gdt):  deprecated. use cython see cython/it_IoU.pyx
  """
  anchor:  num x 1 x 1 x 5 # 5:       x,y,alphaR,rh,rw
                                            0 1    2   3  4

  gdt   :  n  x 6         # 6: lablel, x , y , alphaR , rh , rw
                                  0    1   2      3      4    5

               return an_gdtbox, an_weight, an_label  
           an_gdt:
         ground truth foreach anchor
  """

class GroupMetrix(object):
  """
         for net training
  """
  def __init__(self):
    self.acc = 0.
    self.bbox_loss = 0.
    self.batch_cnt = 0
    self.nd_1 = mx.nd.array([-1,])

  def update(self, mod, d):
    bbox_loss_ , label_score_ = [ mod.get_outputs()[_]  for _ in xrange(2) ]
    bbox_loss  = mx.nd.sum(bbox_loss_).asnumpy()[0]
    pred_label  = mx.nd.argmax(  label_score_ , axis=2)
  
#    logging. info([ pred_label.shape, d.label[0].shape])
    
    acc_1  =mx.nd.sum( pred_label == d.label[0] )
    acc_3  =mx.nd.sum(mx.nd.broadcast_equal(mx.nd.ones_like(d.label[0],context=d.label[0].context) , d.label[0])).asnumpy()[0]
    acc_2  =d.label[0].shape[1]
    acc  = acc_1 / (acc_2-acc_3)
    acc  = acc.asnumpy()[0]
#    assert 0

    self. acc += acc
    self. bbox_loss += bbox_loss
    self. batch_cnt += 1

  def reset(self):
    self. acc = 0.
    self. bbox_loss = 0.
    self. batch_cnt = 0

  def getMetric(self):
    return 'bbox loss: %.6f, acc: %.6f'%(self. bbox_loss/ self.batch_cnt, self. acc/self. batch_cnt)




def proc4pred(imgPath, mean=[0,0,0], std=1):
    """
        return img_data for DataBatch
    """
    with open(imgPath,'rb') as f:
        img = mx.nd.array( mx.img.imdecode(f.read()).asnumpy() - np.array(mean) )/std
    img = mx.nd.transpose(img, axes = (2, 0, 1) )
    img_data = mx.nd.expand_dims(img, axis=0)
    return img_data




