"""
i think it remains a mystery to test whether the net can be used to regress the angle...
but the scarcity is a serious main balk...

test
1.for raw source, i rotate the src and crrop the gdt
2. for generating, i rotate the gdt and add them to the src...
"""
import numpy as np
import mxnet as mx
import cv2, os
import matplotlib.pyplot as plt
from crop import rotation_crop

imgdir='../../data/448'
txtpath= '../../data/angleList-448.txt'

imgname='390.png'


def rotate_img(img, c, deg):
    M= cv2.getRotationMatrix2D(tuple(c),deg,1)
    return cv2.warpAffine(img,M,tuple(2*np.array(img.shape[:2])))


def gen_label_dict(labelfile):
    d= {}
    with open(labelfile,'r') as f:
        for s in f:
            s=s.strip()
            imgname = s.partition(';')[0]
            d[imgname] = s
    return d

def parseLine(imgdir, s, ret_img=False):
    # from angleIter.py

#    lineStr=fs[lineIdx]
    lg = s.rsplit(';\t')
    imgName = lg.pop(0)
    img=mx.image.imdecode(open(os.path.join(imgdir,imgName),'rb').read())


    lgSize=len(lg)   # as Batch Channel
    imgHWC=img.shape
    H,W,C=imgHWC

    #     label, bbx, weight
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
    if ret_img:
        return gdt, img.asnumpy()
    else:
        return gdt

def gen_endian(entry):
    """
        gen top-left and bottom-right points for crop...
        entry:
            l, x,y alpha, rh, rw
                l, apha are ignored...
             {rh} along x-axis
    """
    #pass
    x_tl, x_br = [int(entry[1]+h) for h in [-entry[-2], entry[-2]] ]
    y_tl, y_br = [int(entry[2]+w) for w in [-entry[-1], entry[-1]] ]
    return (x_tl, y_tl), (x_br, y_br)


d = gen_label_dict(txtpath)
s=d[imgname]
img= cv2.imread(os.path.join(imgdir,imgname))
img = img[:,:,[2,1,0]]
gdt = parseLine(imgdir, s)
#print gdt
patch_list = rotation_crop(imgdir, s)
plt.imshow(img)
plt.figure()
plt.imshow(patch_list[1])
plt.show(block=False)
s=raw_input('any key to close...')
assert 0

for entry in gdt[1:2]:# test one...
    print entry
    print np.rad2deg(entry[3])
    r_img = rotate_img(img, entry[1:3], np.rad2deg(entry[3]))
    tl, br= gen_endian(entry)
    # crop...
    patch = r_img[tl[1]:br[1],tl[0]:br[0],  :]

plt.imshow(r_img)
plt.figure()
plt.imshow(img)
plt.figure()
plt.imshow(patch)
plt.title('show a cropped patch')
plt.show(block=False)
s=raw_input('any key to close...')
