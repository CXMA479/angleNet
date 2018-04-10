"""
crop patches,
    paste them to some plain plant...
"""
from crop import rotation_crop, gen_label_dict, pad_rotate, patch_synthesis
import mxnet as mx
import matplotlib.pyplot as plt
import cv2, os
import numpy as np


imgdir='../../data/448'
txtpath= '../../data/angleList-448.txt'
crop_dir = '../../data/crop/'

imgname_list=['390',]

imgname_list = [name+'.png' for name in imgname_list ]

s_dict = gen_label_dict(txtpath)
background_file = 'background-448.png'
background_img = cv2.imread(background_file)[:,:,[2,1,0]]

for imgname in imgname_list[:1]:
    s = s_dict[imgname]
    if ';' not in s:
        continue
    patch_list = rotation_crop(imgdir, s)
    for idx,patch in enumerate(patch_list):
        C= np.random.randint(150, 300,(2,))
        rotated_pad = pad_rotate(patch,30, (448,448),None)#C)
        synthesis = patch_synthesis(rotated_pad, background_img)
#        patch = synthesis#rotated_pad
        cv2.imwrite(os.path.join(crop_dir,str(idx)+'_'+imgname),patch[:,:,[2,1,0]] )






