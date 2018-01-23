import mxnet as mx
import numpy as np
import sys, time, logging
from config import config as cfg
from angleIter import angleIter
from make_symbol import gen_symbol, gen_model, feval_l1_angleMetric
from tool import mytick,viz_score_predict, viz_bbox_gdt, viz_target
import matplotlib.pyplot as plt
cfg.it.debug.debug=  True

symbol_mod, fixed_param_names = gen_symbol()

it=angleIter(symbol_mod)#'../../data/angleList.txt','../../data/img',1,100)
d=it.next()
d0=d
#viz_bbox_gdt(it.raw_img, it.anchor, d0.label[1].asnumpy(), d0.data[2].asnumpy(),show_num=None )


mA_acc=mx.metric.create('acc') #mx.metric.CustomMetric(angleMetric)
mA_l1 = mx.metric.CustomMetric(feval_l1_angleMetric, name='l1_smooth loss')

mod = gen_model(symbol_mod,fixed_param_names,it)
#t= mytick()
#t.load()

#mod.forward(d)
#mod.acc_backward()
#out = mod.get_outputs()
#mA_acc.update([out[0]], [ d.label[0]  ])
#mA_l1.update([out[1]], [ d.label[1]  ])

t0 = time.time()

acc_list =[]
#mod.forward(d0)


#feat_data_0= mod.get_outputs()[2].asnumpy()



out= mod.get_outputs()[0].asnumpy()
out = np.argmax(out, axis=2)[0]
l=d0.label[0].asnumpy()[0].astype(int)
err = out[l!=-1]-l[l!=-1]
acc_list.append( (err==0).sum()*1. / len(err) )

execfile('clip.py')
out= mod.get_outputs()[0].asnumpy()
out = np.argmax(out, axis=2)[0]
l=d0.label[0].asnumpy()[0].astype(int)
err = out[l!=-1]-l[l!=-1]
acc_list.append( (err==0).sum()*1. / len(err) )
#>>> acc_list
#[0.5, 1.0]     seems ok~ 10 is sufficient
# check bounding-box...

fg=plt.figure()
#plt.subplot()

img=viz_target(it.raw_img, it.anchor, d0.label[1].asnumpy(),mod.get_outputs()[1].asnumpy(),\
                         mod.get_outputs()[0].asnumpy(),d0.data[1].asnumpy(),show_num=None)


viz_score_predict(it.raw_img, it.anchor, d0.label[1].asnumpy(),mod.get_outputs()[1].asnumpy(),\
              mod.get_outputs()[0].asnumpy(), score_th=.5,ax=fg.gca(),show_num=20 )

# import matplotlib.pyplot as plt;plt.hist(mod.get_outputs()[2].asnumpy().flatten(),range=(-1,mod.get_outputs()[2].asnumpy().flatten().max()));plt.show()
# mx.nd.sum(mx.nd.abs(mod.get_outputs()[2])).asnumpy()/np.prod(mod.get_outputs()[2]).shape)
#feat_data_1= mod.get_outputs()[2].asnumpy()
out= mod.get_outputs()[0].asnumpy()
out = np.argmax(out, axis=2)[0]
l=d0.label[0].asnumpy()[0].astype(int)
err = out[l!=-1]-l[l!=-1]
acc_list.append( (err==0).sum()*1. / len(err) )





logging.info(d0.label[0])
for epoch_i in xrange(cfg.it.epoch):
  it.reset()
  for batch_i, d in enumerate(it):
    mod.forward(d0)
#    mod.acc_backward()
    mod.backward()
    mod.update()

    out = mod.get_outputs()
    logging.info(out[0].asnumpy() )
#    logging.info(type(out[0]) )
#    logging.info(out[0].asnumpy())
    mA_acc.update([out[0]], [ d0.label[0]  ])
    mA_l1.update([out[1]], [ d0.label[1]  ])

    if batch_i % cfg.train.batch_size ==0 and batch_i is not 0: # update
      pass
#      mod.acc_update(cfg.train.batch_size)

    if batch_i % cfg.train.callbackBatch ==0 and batch_i is not 0: # log info
      t1=time.time()
      elapse = t1-t0
      logging.info('epoch[%d], batch:%d, acc:%f, rpn loss:%f, %.2f samples/sec'%\
          (epoch_i, batch_i,mA_acc.get()[1], mA_l1.get()[1], cfg.train.batch_size*cfg.train.callbackBatch*1./elapse ) )
      mA_acc.reset()
      mA_l1.reset()
      t0 = time.time()
      
  mod.save_checkpoint(cfg.train.save_prefix, epoch_i)
assert 0




label= d.label  #
label = mx.nd.reshape(label[0],(-1,)).asnumpy().astype(int)
label_idx = label==1


gdt_bbox = d.label[1].asnumpy()





in_weight, out_weight = [d.data[i].asnumpy() for i in xrange(1,3) ]
in_weight[0,label_idx][:].shape  #(45, 5)
out_weight[0,label_idx][:].shape # (45, 5)
label_idx.sum()   # 45



mod = gen_model(symbol_mod,fixed_param_names,it)

mod.forward(d)

mod.get_outputs()[0].asnumpy()  # 77.91079712  sum the abs    #  4855633. mask weight


pred_bbox = mod.get_outputs()[1]
pred_bbox.shape
pb = pred_bbox.asnumpy()
pb[0,label_idx][:].max()    # 1.0759475
#1.0759475

pred_bbox = mod.get_outputs()[1]
pred_bbox.shape
pb = pred_bbox.asnumpy()
pb[0,label_idx][:].max()  #78.950371




####   A number of loops...   #
mod.forward(d)
#mod.get_outputs()[0].asnumpy().max()
abs(mod.get_outputs()[0].asnumpy()[0,label_idx][:]).sum()
mod.acc_backward()
mod.forward(d)
mod.acc_backward()
mod.acc_update(2)


pred_bbox = mod.get_outputs()[0]
pred_bbox.shape
pb = pred_bbox.asnumpy()
pb[0,label_idx][:].max()  #3.8684198e+09
# 3.1520758e+10
#   after blockgrad of the score, still -> 3688756.5(some loops)
#       using abs(-), still: 6.2079197e+10

abs(pb[0,label_idx][:] - d.label[1].asnumpy()[0,label_idx][:]).sum()


pred_bbox = mod.get_outputs()[2]
pred_bbox.shape
pb = pred_bbox.asnumpy()
pb[0,label_idx][:].max()    # 1.8828602e+13


mod.get_outputs()[0].asnumpy()   #   2.34176528e+13

mod.save_checkpoint(cfg.train.save_prefix, 0)


label= d.label  #
l = label[1].asnumpy()
l.max()
#0.73612159



d = data[2].asnumpy()
d.max()
#0.013333334

it.provide_label
#[('target_label', (1L, 37800L)), ('target_bbox', (1L, 37800L, 5L))]
it.provide_data
#[('data', (1L, 3L, 800L, 1000L)), ('rpn_inside_weight', (1L, 37800L, 5L)), ('rpn_outside_weight', (1L, 37800L, 5L))]

for epoch_i in enumerate(cfg.it.epoch):
  mod.forward(d)
  mod.acc_backward()
  mod.acc_update()

