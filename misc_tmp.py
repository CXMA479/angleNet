import numpy as np
import os, time
import cPickle as cpk
time_stamp=time.asctime().replace(' ','-')
save_dir='../data/debug/'
with open(os.path.join(save_dir, time_stamp)+'.cpk','w') as f:
    d={}
    d['anchor'] = it.anchor
    d['d0'] = d0
    d['model_outputs'] = mod.get_outputs()
    cpk.dump(d,f)

"""
with open(os.path.join(save_dir, time_stamp)+'.cpk','r') as f:
    d=cpk.load(f)

"""
