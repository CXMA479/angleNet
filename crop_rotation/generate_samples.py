"""
generate pics and txt from single specific patch and background image.
"""
from crop import rotation_crop, gen_label_dict, pad_rotate, patch_synthesis
import mxnet as mx
import matplotlib.pyplot as plt
import cv2, os
import numpy as np

crop_file = '1_390.png'
background_file='background-448.png'
imgoutput_dir='../../data/synthesis/test_2_390/448'
location_bound= (150,300)

if not os.path.isdir(imgoutput_dir):
    os.mkdir(imgoutput_dir)
txtfile = os.path.join( os.path.dirname(imgoutput_dir), 'angleList-'+os.path.basename(imgoutput_dir)+'.txt' )
num_sample = 5000
patch = cv2.imread(crop_file)
h,w = patch.shape[:2]
background_img = cv2.imread(background_file)

assert max(location_bound) < min(background_img.shape[:2]), (location_bound, background_file)
with open(txtfile,'w') as f:
    entry =''
    for sample_idx in xrange(num_sample):
        angle = np.random.randint(0,180)
        imgname = '%d.png'%sample_idx
        x,y=np.random.randint(location_bound[0], location_bound[1],(2,))
        entry += '%s;\t1\t%d,%d\t%d\t%d\t%d\n'%(imgname,x,y,angle,w/2,h/2)  # NTOE! rh is not h/2 !
        rotated_pad = pad_rotate(patch, angle, background_img.shape[:2], (x,y))

        sample = patch_synthesis( rotated_pad, background_img)
        cv2.imwrite(os.path.join(imgoutput_dir,imgname), sample)
    f.write(entry.strip())









