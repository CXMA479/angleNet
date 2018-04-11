"""
derived from mx-rcnn/rcnn/processing/nms.py

we need a clean figure -_-|

Chen Y.liang

"""
import mxnet as mx
from gpu_IoU import gpu_it_IoU
from tool import bbox_inv_transfer
import numpy as np
import time
#sad

def mx_nms(raw_pred, anchor, score, box=None,\
            iou_thresh=.5, score_thresh=.9, min_area=50*50,max_area=100*200,min_length=30,max_length=220,ctx=mx.gpu()):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    args:
        box:    mx.NDArray  [[x, y, alpha_R, rh, rw]],  (n, 5)
        score:  mx.NDArray  (n,xxx, 2)
        thresh: retain overlap < thresh
    return: indexes to keep
    """
#    print score.shape
    score = score.asnumpy()[0] if not isinstance(score,np.ndarray) else score
    score = score[:,1]
    assert len(score.shape)==1, score.shape
    raw_pred = raw_pred[0]
#    assert 0, (anchor.shape, raw_pred.shape)
    box = bbox_inv_transfer(anchor, raw_pred.asnumpy()) if anchor is not None else box
    dynamic_score_list = np.array(list(score.asnumpy())) if not isinstance(score,np.ndarray) else list(score)
    dynamic_score_idx_list= list(xrange(len(dynamic_score_list))) # recording org idx
    sub_box = box.copy() if isinstance(box, np.ndarray) else box.asnumpy()# np.array would be better 2 idx
    sub_score = np.array(dynamic_score_list[:])
    sub_idx= np.array(dynamic_score_idx_list[:])

    # filter with score_threshold...
    ps_score_idx = sub_score > score_thresh

    sub_score  = sub_score[ps_score_idx]
    sub_box    =   sub_box[ps_score_idx]
    sub_idx    = sub_idx[ps_score_idx]
#    print('%d boxes to proc...'%len(sub_idx))
    # filter area...
    areas = 4*sub_box[:,3]*sub_box[:,4]
    ps_area_idx = (areas>min_area) * (areas<max_area)
    sub_score  = sub_score[ps_area_idx]
    sub_box    =   sub_box[ps_area_idx]
    sub_idx    = sub_idx[ps_area_idx]

    # filter length...
    min_l = 2*sub_box[:,3:].min(axis=-1)
    max_l = 2*sub_box[:,3:].max(axis=-1)
    ps_area_idx = (min_l>min_length) * (max_l<max_length)
    sub_score  = sub_score[ps_area_idx]
    sub_box    =   sub_box[ps_area_idx]
    sub_idx    = sub_idx[ps_area_idx]

    print('%d boxes to proc...'%len(sub_idx))
    keep = []
    while len(sub_idx) >0:
        max_sub_idx = sub_score.argmax()
        keep.append(sub_idx[max_sub_idx])
        iou_table = gpu_it_IoU(sub_box[max_sub_idx:max_sub_idx+1],sub_box, k=10, ctx=ctx)[0]

        lefted_sub_idx = iou_table < iou_thresh
        print(time.asctime(), np.sum(lefted_sub_idx))
        if np.sum(lefted_sub_idx) == 0:
            return keep
        sub_box = sub_box[lefted_sub_idx,:]
        sub_score = sub_score[lefted_sub_idx]
        sub_idx = sub_idx[lefted_sub_idx]
    return keep





    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep







