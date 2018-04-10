import mxnet as mx
import numpy as np
import logging
import sys,os, time, cv2
import matplotlib.pyplot as plt

def gen_label_dict(txtpath):
    d={}
    with open(txtpath,'r') as f:
        for s in f:
            if ';' in s:
                s=s.strip()
                d[s.partition(';')[0]]=s
    return d

def parseLine(imgdir, s, ret_img=False):
    # from angleIter.py

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

def rotate_img(img, c, deg, shape_mult=2):
    M= cv2.getRotationMatrix2D(tuple(c),deg,1)
    return cv2.warpAffine(img,M,tuple(shape_mult*np.array(img.shape[:2])))
def pad_rotate(patch, deg, HW, C=None):
    """
        HW: (H,W) output shape
    """
    cy, cx = [z/2 for z in patch.shape[:2]] # in patch
    CY, CX = [z/2 for z in HW] if C is None else C[::-1]  # in output shape
    top_pad = CY-cy
    bottom_pad = (HW[0]-CY) - (patch.shape[0]-cy)
    left_pad = CX-cx
    right_pad = (HW[1]-CX) - (patch.shape[1]-cx)
#    assert 0, (patch.shape, top_pad, bottom_pad)

    pad = np.pad(patch,((top_pad, bottom_pad),(left_pad, right_pad),(0,0)), 'constant', constant_values=0)
    # rotate...
    return rotate_img(pad, (CX,CY),deg, shape_mult=1)

def patch_synthesis(patch, img):
    sign = 1- np.sign(patch)
    return patch+img*sign

def crop_img(img, tl, br, ignore_partitial=True):
    assert img.shape[2]==3, img.shape
    H,W = img.shape[:2]
    if ignore_partitial:# drop incomplete patches...
        if min(tl)<0 or br[0]>W or br[1] > H:
            return None
    tl_x, tl_y = [ max(0, tl_z) for tl_z in tl ]
    br_x, br_y = [min(R, br_z) for R, br_z in zip([W,H], list(br)) ]
#    print tl, br
#    print (tl_x, br_x, W), (tl_y, br_y, H)
#    assert 0
    return img[tl_y:br_y,tl_x:br_x,  :]

def gen_endian(entry):
    x_tl, x_br = [int(entry[1]+h) for h in [-entry[-2], entry[-2]] ]
    y_tl, y_br = [int(entry[2]+w) for w in [-entry[-1], entry[-1]] ]
    return (x_tl, y_tl), (x_br, y_br)


def rotation_crop(imgdir,line_s):
    """
        img: raw iamge, np.NDArray from cv2.read
        return:
            a list of patches, croped from the image with the line_s
    """
    if ';' not in line_s:
            return []
    entries, img = parseLine(imgdir, line_s, ret_img=True)
    patch_list = []
#    print entries
    for entry in entries:
        r_img = rotate_img(img, entry[1:3], np.rad2deg(entry[3]))
        tl, br= gen_endian(entry)
        patch =crop_img(r_img, tl, br)
        if patch is not None:
#            print tl,br, patch.shape
#            assert 0
            patch_list.append(patch )

    return patch_list


