import mxnet as mx
from predict_viewer import Viewer
import config, os
config.nofile()

model_prefix='Apr-9/Mon Apr  9 19:44:07 2018_angleD=[0];HoW=[2, 0.5, 1];sideLength=[80, 60]';epoch=2#'Mon Apr  9 07:59:48 2018_angleD=[0];HoW=[2, 0.5, 1];sideLength=[80, 120, 60]';epoch=0#'Apr-8/Sun Apr  8 17:29:25 2018_angleD=[0, 90];HoW=[2, 0.5, 1];sideLength=[80, 120, 60]';epoch=0;#'Sun Apr  8 16:26:07 2018_angleD=[0];HoW=[2, 0.5, 1];sideLength=[80, 120, 60]';epoch=0#'Sun Apr  8 14:50:23 2018_angleD=[0];HoW=[2, 0.5, 1];sideLength=[80, 120, 60]';epoch=0#'Sun Apr  8 10:13:05 2018_angleD=[0, 60, -60];HoW=[2, 0.5, 1];sideLength=[80, 120, 60]';epoch=1#Apr-5/Thu Apr  5 09:18:19 2018_angleD=[0, 60, -60];HoW=[2, 0.5, 1];sideLength=[80, 120, 60]';epoch=2#'Apr-4/Wed Apr  4 21:10:48 2018_angleD=[0, 60, -60];HoW=[2, 0.5, 1];sideLength=[80, 120, 60]';epoch=6#'Apr-4/Wed Apr  4 15:06:06 2018_angleD=[0, 60, -60];HoW=[2, 0.5, 1];sideLength=[80, 120, 60]';epoch=0#'Mar-6/Tue Mar  6 10:33:58 2018_angleD=[0, 60, -60];HoW=[2, 0.5, 1];sideLength=[80, 120, 60]';epoch=3#'Apr-4/Wed Apr  4 08:55:07 2018_angleD=[0, 60, -60];HoW=[2, 0.5, 1];sideLength=[80, 120, 60]';epoch=2#''Apr-3/Tue Apr  3 20:10:44 2018_angleD=[0, 60, -60];HoW=[2, 0.5, 1];sideLength=[80, 120, 60]';epoch=0#'Tue Apr  3 15:52:34 2018_angleD=[0, 60, -60];HoW=[2, 0.5, 1];sideLength=[80, 120, 60]';epoch=1#Mar-22/Thu Mar 22 22:19:43 2018_angleD=[0, 60, -60];HoW=[2, 0.5, 1];sideLength=[80, 120, 60]';epoch = 2#'Mar-29/Thu Mar 29 08:47:05 2018_angleD=[0, 60, -60];HoW=[2, 0.5, 1];sideLength=[80, 120, 60]';epoch=0#'Sun Mar 11 09:33:55 2018_angleD=[0, 60, -60];HoW=[2, 0.5, 1];sideLength=[80, 120, 60]';epoch = 0#Mar-10/Sat Mar 10 21:15:39 2018_angleD=[0, 60, -60];HoW=[2, 0.5, 1];sideLength=[80, 120, 60]';epoch=0#

v = Viewer(os.path.join('../output/',model_prefix),epoch)
imgname='man-0.png'#'../data/448/2.png'#'../data/test/448/t8.png'#'448.png'#

v.predict(imgname)
v.view(0.5,.5)
v.filter_transfered_bbox
v.predict_transfered_bbox
v.predict_score
s=raw_input('print any key to close...')
