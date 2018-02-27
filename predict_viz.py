"""
load trained model, predict and then, viz...
"""

import mxnet as mx
import numpy as np
import sys, time, logging, os
from config import config as cfg
from angleIter import angleIter
from make_symbol import gen_symbol, gen_model, feval_l1_angleMetric
from tool import mytick,viz_score_predict, viz_bbox_gdt, viz_target
import tool, cv2
import matplotlib.pyplot as plt


############################################
####    choose one block to fill  ##########
###############   BLOCK I      #############
labelFile =None
imgdir    =None
###############   BLOCK II      #############
"""  this block refers to bare image reading without label file..."""
imgPath   ='51.png'
#############################################


ctx = mx.gpu()

model_prefix ='../output/Sat Jan 27 14:18:51 2018_angleD=[0, 60, -60];HoW=[1.5, 3];sideLength=[60, 80]'#Sat Jan 27 10:12:11 2018_angleD=[0, 60, -60];HoW=[1.5, 3];sideLength=[60, 80]'
model_epoch =6#0

symbol, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, model_epoch)
symbol_pred = mx.sym.Group(   [symbol.get_internals()[x+'_output'] for\
                     x in ['rpn_label_loss', 'rpn_pred_bbox_reshape'] ] )
                     
mod = mx.mod.Module(symbol_pred, data_names=('data',), label_names=('target_label',), context=ctx)





# need an anchor generator...
#   for raw image prediction use.
if os.path.isfile(imgPath):
    feat_symbol = symbol.get_internals()[cfg.net.rpn_conv_name+'_output']
    with open(imgPath,'rb') as f:
        img_mx_raw =mx.img.imdecode(f.read())
        img = mx.nd.array( img_mx_raw.asnumpy() - np.array(cfg.it.mean) )
    H,W,C = img.shape
    _, feat_shape, _ = feat_symbol.infer_shape(data=(1,C,H,W))
    feat_shape = feat_shape[0]
    im_info = np.array( [H,W] )
    stride=(im_info[0]/feat_shape[-2],im_info[1]/feat_shape[-1])
    anchor = tool.genAnchor( im_info,cfg.ANCHOR_angleD, cfg.ANCHOR_HoW, cfg.ANCHOR_sideLength,\
              feat_shape, stride) # anchor : N x 5
    img = mx.nd.transpose(img, axes = (2, 0, 1) )
    img_data = mx.nd.expand_dims(img, axis=0)
    dataBatch = mx.io.DataBatch([img_data,])
    mod.bind(data_shapes=[ ('data',img_data.shape), ], for_training=False)
    mod.init_params(None, arg_params=arg_params, aux_params= aux_params, allow_missing=False,\
                  allow_extra=True)

    mod.forward(dataBatch)
    viz_score_predict(img_mx_raw.asnumpy().astype(np.uint8), anchor, None,mod.get_outputs()[1].asnumpy(),\
              mod.get_outputs()[0].asnumpy(), score_th=.1,show_num=150 )
    mx.nd.waitall()
#    print 'pass test succeeded!'
#    assert 0
else:
    it=angleIter(symbol_mod, is_train=False,\
            labelFile=labelFile, imgPath=imgPath)# use data/iou/$scale/test/
    d=it.next()
#d0=d
#viz_bbox_gdt(it.raw_img, it.anchor, d0.label[1].asnumpy(), d0.data[2].asnumpy(),show_num=None )

mod = gen_model(symbol_mod,fixed_param_names,it)
fg=plt.figure()

img=viz_target(it.raw_img, it.anchor, d0.label[1].asnumpy(),mod.get_outputs()[1].asnumpy(),\
                         mod.get_outputs()[0].asnumpy(),d0.data[1].asnumpy(),show_num=None)


viz_score_predict(it.raw_img, it.anchor, d0.label[1].asnumpy(),mod.get_outputs()[1].asnumpy(),\
              mod.get_outputs()[0].asnumpy(), score_th=.5,ax=fg.gca(),show_num=20 )


