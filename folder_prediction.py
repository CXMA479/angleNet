import os, cv2, config
import numpy as np
from predict_viewer import Viewer
config.nofile()

model_prefix='Fri Apr 20 09:04:32 2018_angleD=[0, 60, -60];HoW=[2, 0.5, 1];sideLength=[120, 80, 160]';epoch=7

"""
'Wed Jun 13 10:50:53 2018_angleD=[0];HoW=[2, 0.5, 1];sideLength=[64, 128, 256]';epoch=11#
'Wed Jun 13 10:50:53 2018_angleD=[0];HoW=[2, 0.5, 1];sideLength=[64, 128, 256]';epoch=7#
'Wed Apr 11 10:11:49 2018_angleD=[0];HoW=[2, 0.5, 1];sideLength=[80, 60, 100]';epoch=6#
'Wed Apr 11 10:11:49 2018_angleD=[0];HoW=[2, 0.5, 1];sideLength=[80, 60, 100]';epoch=3#
'Thu Apr 19 11:04:11 2018_angleD=[0, 60, -60];HoW=[2, 0.5, 1];sideLength=[90, 60, 130]';epoch=7#
'Sun Apr 22 11:58:35 2018_angleD=[0];HoW=[2, 0.5, 1];sideLength=[90, 60, 130]';epoch=19#
'Sat Apr 21 16:03:03 2018_angleD=[0];HoW=[2, 0.5, 1];sideLength=[120, 80, 160]';epoch=19#
'Fri Apr 20 09:04:32 2018_angleD=[0, 60, -60];HoW=[2, 0.5, 1];sideLength=[120, 80, 160]';epoch=19#
"""

FOR_BOX = not True # only switch to save boxes, not visual results

size_i = 448*2#'896'#
image_dir = '../data/metric/%d'%size_i
save_dir = '../res/%d'%size_i + model_prefix+'-'+str(epoch)
box_save_dir = '../output/metric-model/AG/%d-Apr-19/boxes'%size_i


assert os.path.isdir(image_dir), image_dir
for d in [save_dir, box_save_dir]:
  if not os.path.isdir(d):
    print 'create directory[%s]'%d
    os.mkdir(d)

v = Viewer(  os.path.join('../output/metric-model', model_prefix), epoch  )
img_lst = os.listdir(image_dir)
for imgname in img_lst:
  v.predict( os.path.join(image_dir, imgname) )
  if not FOR_BOX:
    v.view(0.8, 0.05, quiet_model=True)
    cv2.imwrite( os.path.join(save_dir, imgname), v.view_img[:,:,[2,1,0]]  )
  else:
    # save to boxes...
    boxes = v.view(0.8, 0.05, onlyneed_box=True)
    txtpath = os.path.join(box_save_dir,\
                'result-boxes-%s.txt'%imgname)
    np.savetxt(txtpath, boxes, delimiter=',', \
                fmt='%d,%d,%.4f,%d,%d,%.4f ')

