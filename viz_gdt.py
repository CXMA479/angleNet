import mxnet as mx
import numpy as np
import sys, time, logging, os, cv2
import config
config.nofile()
cfg = config.config
from angleIter import angleIter
from make_symbol import gen_symbol, gen_model, feval_l1_angleMetric
from tool import draw_angleBox
import matplotlib.pyplot as plt
cfg.it.debug.debug=  True

imgdir='../data/448'
labeltxt= '../data/angleList-448.txt'
outputdir='../data/showgdt/448'

if not os.path.isdir(outputdir):
    os.mkdir(outputdir)


def gen_label_dict(labelfile):
    d= {}
    with open(labelfile,'r') as f:
        for s in f:
            s=s.strip()
            imgname = s.partition(';')[0]
            d[imgname] = s
    return d


def parseLine(s):
    """
        Only for decoding string for obtaining x,y,alpha,rh,rw
    """
    # from angleIter.py

#    lineStr=fs[lineIdx]
    lg = s.rsplit(';\t')
    imgName = lg.pop(0)
#    img=mx.image.imdecode(open(os.path.join(imgdir,imgName),'rb').read())
#    ret_img = img.asnumpy().astype(np.uint8)


    lgSize=len(lg)   # as Batch Channel
#    imgHWC=img.shape
#    H,W,C=imgHWC
#    if feat_sym is not None:
#      _, feat_shape, _=feat_sym.infer_shape(**{cfg.it.dataNames[0]:(1,C,H,W)})  # it only needs img
#      feat_shape = feat_shape[0]

    #     label, bbx, weight
#    feat_shape = feat_shape
    gdt=np.zeros((lgSize,6))
#    im_info=np.array([H,W])

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
    return gdt

d = gen_label_dict(labeltxt)
imglist = os.listdir(imgdir)
for imgname in imglist:
    s=d[imgname]
    if ';' not in s:
        print('invalid image[%s], skip...'%imgname)
        continue
    img = cv2.imread(os.path.join(imgdir,imgname) )
    gdts= parseLine(s)
#    print (gdts)
#    assert 0
    img = draw_angleBox(img,gdts[:,1:].astype(np.float), [255,0,0], line_width=2)
    cv2.imwrite(os.path.join(outputdir, imgname), img)


print('Save Ground Truth into %s'%outputdir)
