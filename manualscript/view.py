from feat_viewer import Viewer
import os, cv2
import matplotlib.pyplot as plt
AG_model = '../../output/metric-model/Thu Apr 19 11:04:11 2018_angleD=[0, 60, -60];HoW=[2, 0.5, 1];sideLength=[90, 60, 130]';epoch=7

#AG_model = '../../model/vgg16';epoch=0
imgdir='.'#./../data/metric/manualscript/'
imgname='man2.png'

"""
v = Viewer(AG_model,epoch, mean=[123.68, 116.28, 103.53])
v.load_layers( 'relu5_3',bind_size=(224,224) )
v.predict( os.path.join(imgdir, imgname) ,re_size_HW=(224, 224) )

v.view()

s=raw_input('Press S to save raw featureMap\n')
if s=='S':
    v.save_rawfeat('feature-%s'%imgname)#v.save_rawfeature('feature-%s'%imgname)
#s=raw_input('press any key to exit')
"""

##################################
#   draw anchor
##################################
current_dir = os.getcwd()
os.chdir('..')
from tool import genAnchor, draw_angleBox
os.chdir(current_dir)

imgsize = 224
feat_size =14 *4
stride = imgsize/ feat_size
stride = [ imgsize,  imgsize ]
feat_shape = [ feat_size, feat_size]
angleD=[0, 60, -60]
HoW=[2, 0.5, 1]
sideLength=[90, 60, 30]

img = cv2.imread( os.path.join(imgdir, imgname) )[:,:,[2,1,0]]
anchor = genAnchor(None, angleD, HoW, sideLength, feat_shape, stride)

img_anchor = draw_angleBox(img, anchor, [0, 0, 255])
plt.imshow(img_anchor)
plt.show()
