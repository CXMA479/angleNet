"""
this scipt tends to check gdt by
  drawing them on the channels.
"""

from iouProb_metric import gen_label_dict, parseLine
from tool import draw_angleBox
import matplotlib.pyplot as plt

if __name__ == '__main__':
  size_i = 896
  target_img = '34.png'

  imgdir='../data/metric/%d'%size_i; labelfile='../data/metric/angleList-%d.txt'%size_i # contents will be analysed...
  labelfile = '../data/angleList-%d.txt'%size_i
  d = gen_label_dict(labelfile)
  s = d[target_img]
  im_info, feat_shape, gdt, img = parseLine(s, imgdir)
  img = draw_angleBox(img, gdt[:,1:], (0,0,255))
  plt.imshow(img)
  plt.show()



