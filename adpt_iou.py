"""
aims to adapt a suit set of parameters for IoU threshold
"""

import mxnet as mx
import numpy as np
import tool, config, os
config.nofile()
cfg = config.config
from gpu_IoU.gpu_IoU import gpu_it_IoU
import matplotlib.pyplot as plt

imgdir='../data/448'; labelfile='../data/angleList-448.txt' # contents will be analysed...

assert os.path.isdir(imgdir), imgdir
assert os.path.isfile(labelfile), labelfile


def gen_label_dict(labelfile):
    d= {}
    with open(labelfile,'r') as f:
        for s in f:
            s=s.strip()
            imgname = s.partition(';')[0]
            d[imgname] = s
    return d

imglist = os.listdir(imgdir)
label_dict = gen_label_dict(labelfile)


symbol = mx.sym.load('%s-symbol.json'%cfg.net.symbol_path)
feat_sym=symbol.get_internals()[cfg.net.rpn_conv_name+'_output']  if cfg.it.debug.feat_shape is None else None
iou_list =[]

def parseLine(s):
    # from angleIter.py

#    lineStr=fs[lineIdx]
    lg = s.rsplit(';\t')
    imgName = lg.pop(0)
    img=mx.image.imdecode(open(os.path.join(imgdir,imgName),'rb').read())
    img = mx.nd.array( img.asnumpy())


    lgSize=len(lg)   # as Batch Channel
    imgHWC=img.shape
    H,W,C=imgHWC
    if feat_sym is not None:
      _, feat_shape, _=feat_sym.infer_shape(**{cfg.it.dataNames[0]:(1,C,H,W)})  # it only needs img
      feat_shape = feat_shape[0]

    #     label, bbx, weight
    feat_shape = feat_shape
    gdt=np.zeros((lgSize,6))
    im_info=np.array([H,W])

    try:
        lg.append(lg.pop(-1)[:-1])  # get rid of '\n'
    except:
        assert 0, (lg, imgname)
    #   go for 'label', 'gdt'
    for i,obj in enumerate(lg):
      """
                                 now obj:  label  x,y  alpha(deg)  rh  rw
      """
      label,xy,alphaD,rh,rw=obj.split('\t')
      label,alphaD,rh,rw = [ np.float(_) for _ in (label,alphaD,rh,rw)]
      x,y=[np.float(_) for _ in xy.rsplit(',') ]
      alpha=np.deg2rad(alphaD)  # deg 2 rad
      gdt[i][:] = np.array([label,x,y,alpha,rh,rw])
      return im_info, feat_shape, gdt

valid_imgcnt=0
for imgname in imglist:
    s = label_dict[imgname]
    if ';' not in s:
        continue
    im_info, feat_shape, gdt = parseLine(s)
    stride=(im_info[0]/feat_shape[-2],im_info[1]/feat_shape[-1])
    anchor = tool.genAnchor(im_info, cfg.ANCHOR_angleD, cfg.ANCHOR_HoW, cfg.ANCHOR_sideLength,\
                feat_shape, stride)
    iou_matrix = gpu_it_IoU(anchor, gdt[:,1:], cfg.train.iou_x_num )
    iou_list += list(iou_matrix.ravel())
    valid_imgcnt += 1

# plot histogram...
title = 'sideLength: %s, #anchor: %d, #valid-img: %d'%(str(cfg.ANCHOR_sideLength), len(iou_list), valid_imgcnt)

hist_range = np.arange(0,1,.1)

plt.hist(iou_list, hist_range)
plt.title(title)
plt.show()













