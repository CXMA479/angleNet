"""
i call it Single Shoot train...

that is, cls will be replaced by reg_cfg and optimized by interacte iteration.

use gluon...

Chen Y. Liang
"""
import mxnet as mx
import numpy as np
from config import config as cfg



import sys, time, logging
from angleIter import angleIter
from make_model import gen_feat_model, gen_bbx_model, gen_iou_model
from tool import mytick,viz_score_predict, viz_bbox_gdt, viz_target
import matplotlib.pyplot as plt







