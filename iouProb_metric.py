"""
for iou metric
"""

import mxnet as mx
import numpy as np
import tool, config, os, time, cv2
config.nofile()
cfg = config.config
from gpu_IoU.gpu_IoU import gpu_it_IoU
import matplotlib.pyplot as plt
from tool import mytick,viz_score_predict, viz_bbox_gdt, viz_target, viz_anchor_byIoU
from predict_viewer import Viewer
from crop_rotation.crop import rotate_img, gen_endian

model_prefix='../output/metric-model/Wed Apr 11 10:11:49 2018_angleD=[0];HoW=[2, 0.5, 1];sideLength=[80, 60, 100]';epoch=6
#model_prefix='../output/metric-model/Thu Apr 19 11:04:11 2018_angleD=[0, 60, -60];HoW=[2, 0.5, 1];sideLength=[90, 60, 130]';epoch=7
#imgdir='../data/metric/448'; labelfile='../data/metric/angleList-448.txt' # contents will be analysed...
imgdir='../data/metric/896'; labelfile='../data/metric/angleList-896.txt' # contents will be analysed...
baseline_dir = '../output/metric-model/baseline'  # store prediction files from baseline
AA_output_prefix = os.path.join( os.path.dirname(labelfile)+'/iou-prob', 'AA_model')
AG_output_prefix = os.path.join( os.path.dirname(labelfile)+'/iou-prob','AG_model'+\
                            os.path.basename(model_prefix) +str(epoch) )

for dir_ele in [AA_output_prefix, AG_output_prefix]:
    dir_ele = os.path.dirname(dir_ele)
    if not os.path.isdir(dir_ele):
        print('Create director: %s...'%dir_ele)
        os.mkdir(dir)

assert os.path.isdir(imgdir), imgdir
assert os.path.isfile(labelfile), labelfile
V=Viewer(model_prefix, epoch)

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
    ret_img = img.asnumpy().astype(np.uint8)


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
    return im_info, feat_shape, gdt, ret_img#, img.asnumpy()


###############       Module I: single image tuning...
# TODO: view anchors and ground-truth in image...

# pick one
#imgidx = np.random.randint(0,len(imglist))
#picked_imgname = None#imglist[imgidx]#'238.png'#
#picked_imgname = '324.png'
#raw_img=mx.image.imdecode(open(os.path.join(imgdir,picked_imgname),'rb').read()).asnumpy() if picked_imgname is not None else None
#assert 0, (raw_img.shape, picked_imgname)

###############       Module II: image directory processing...
valid_imgcnt = 0
save_s='Time:\t%s\nGenerate File:\t%s\nModel:\t%s\nEpoch:\t%d\n'%(time.asctime(), __file__, \
                                os.path.basename(model_prefix), epoch)+'-'*40+'\n'

baseline_save_s= '\n'+'-'*40+'\n'+'Axis-aligned model...\n'+'-'*40+'\n'
# Ok,   I really do not know where I got the current data of axis-aligned method...
## so,,, need a another trail?
### filename template: result-boxes{ImageName}.txt
for imgname in imglist:#['1.png',]:#
    baseline_bbox = np.loadtxt( os.path.join(baseline_dir,'result-boxes%s.txt'%(imgname.split('.')[0])), delimiter=',' )
#    print (os.path.join(baseline_dir,'result-boxes%s.txt'%(imgname.split('.')[0])),baseline_bbox.shape)
#    assert 0
    V.predict(os.path.join(imgdir, imgname) )
    predict_bbox = V.view(.8, .1, onlyneed_box=True)
    s = label_dict[imgname]
    if ';' not in s:
        continue
    im_info, feat_shape, gdt, img = parseLine(s)
    iou_matrix = gpu_it_IoU( predict_bbox[:,:-1], gdt[:,1:], cfg.train.iou_x_num )
    
    baseline_iou_matrix = gpu_it_IoU(baseline_bbox[:,:-1], gdt[:,1:], cfg.train.iou_x_num)
    # concat iou with score
    iou = iou_matrix.max(-1)
    baseline_iou = baseline_iou_matrix.max(-1)
    iou_prob = np.array([iou,predict_bbox[:,-1]]).T
    baseline_iou_prob = np.array([baseline_iou,baseline_bbox[:,-1]]).T
    # dump to file...
    iou_file = AG_output_prefix+'-'+imgname+'.txt'
    baseline_iou_file = AA_output_prefix+'-'+imgname+'.txt'
    np.savetxt(iou_file, iou_prob,header=save_s)
    np.savetxt(baseline_iou_file, baseline_iou_prob,header=save_s)
    continue
    iou = iou_matrix.max(axis=-2) # -2 checks ignore count
    baseline_iou = baseline_iou_matrix.max(axis=-2)
    save_s += 'IoU[%s]:\n'%imgname
    baseline_save_s += 'BASELINE IoU[%s]:\n'%imgname
    baseline_save_s += str(baseline_iou)
    save_s +=str(iou)
    save_s += '\navg:%f\n'%iou.mean()
    baseline_save_s += '\navg:%f\n'%baseline_iou.mean()
    continue
print('store files to directory>>>%s'%os.path.dirname(iou_file) )

"""
crop_output_dir = os.path.join( os.path.dirname(output_file),'crop')
if not os.path.isdir(crop_output_dir):
    os.mkdir(crop_output_dir)
print (img.shape)
for idx, entry in enumerate(predict_bbox):
#    print (entry )
    r_img = rotate_img(img, entry[:2], np.rad2deg(entry[2]))[:,:,[2,1,0]]
    tl, br= gen_endian( entry)
#    print(tl,br)
#    continue
    patch = r_img[tl[1]:br[1],tl[0]:br[0],  :]
    cv2.imwrite(os.path.join(crop_output_dir,'%d.png'%idx), patch )


print (save_s)
print (baseline_save_s)
print('save report to %s...'%output_file)
with open(output_file,'w') as f:
    f.write(save_s[:-1]+baseline_save_s[:-1])
print('Done.')
"""

