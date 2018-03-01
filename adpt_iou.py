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
from tool import mytick,viz_score_predict, viz_bbox_gdt, viz_target, viz_anchor_byIoU

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
        lg.append(lg.pop(-1).strip())  # get rid of '\n'
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
    return im_info, feat_shape, gdt#, img.asnumpy()


###############       Module I: single image tuning...
# TODO: view anchors and ground-truth in image...

# pick one
imgidx = np.random.randint(0,len(imglist))
picked_imgname = None#imglist[imgidx]#'238.png'#
#picked_imgname = '324.png'
raw_img=mx.image.imdecode(open(os.path.join(imgdir,picked_imgname),'rb').read()).asnumpy() if picked_imgname is not None else None
#assert 0, (raw_img.shape, picked_imgname)

###############       Module II: image directory processing...

valid_imgcnt=0
for imgname in imglist:
    if picked_imgname is None or imgname == picked_imgname:
        pass
    else:
        continue
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
    if picked_imgname is not None and imgname == picked_imgname:
        # disp anchors whose IoU > th...
#        assert 0, (gdt, picked_imgname)
#        print 'imgname: %s'%imgname, gdt[:,1:]
        viz_anchor_byIoU(raw_img, anchor, gdt[:,1:], iou_matrix, .7, 'imgname:%s'%imgname)

if picked_imgname is not None:
    assert 0
# plot histogram...
title = 'sideLength: %s, #anchor: %d, #valid-img: %d'%(str(cfg.ANCHOR_sideLength), len(iou_list), valid_imgcnt)

hist_range = np.arange(0,1,.1)

plt.hist(iou_list, hist_range)
plt.title(title)
plt.show()













