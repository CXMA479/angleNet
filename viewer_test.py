import mxnet as mx
from predict_viewer import Viewer
import config
config.nofile()
v = Viewer('../output/Wed Sep 13 15:12:31 2017_angleD=[0, 60, -60];HoW=[1.5, 3];sideLength=[60, 80]',0)
v.predict('0.jpg')
v.view(0.99)
v.filter_transfered_bbox
v.predict_transfered_bbox
v.predict_score
