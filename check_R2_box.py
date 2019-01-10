from tool import draw_angleBox
import os, cv2
import numpy as np
import matplotlib.pyplot  as plt

HOME='/disk1/c20160802009T/'

boxtxt_dir = HOME+'angleNet/metric/R2CNN_FPN_Tensorflow/tools/angleNet_res/896ckpt80000/boxes'
img_dir = HOME+'angleNet/local/data/test/896'
img_name = '34.png'
boxtxt_name = 'result-boxes%s.txt'%(img_name.split('.')[0])


img_path = os.path.join(img_dir, img_name)
assert os.path.isfile(img_path), img_path
img = cv2.imread( img_path  )[:,:,[2,1,0]]

txt_path = os.path.join( boxtxt_dir, boxtxt_name )
assert os.path.join(txt_path), txt_path
boxes = np.loadtxt(txt_path)

img = draw_angleBox(img, boxes[:,:-1], (255,0,0))

plt.imshow(img)
plt.show()
"""
  !TEST OK
"""






