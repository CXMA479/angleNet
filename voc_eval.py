"""
1. based on R2CNN(https://github.com/yangxue0827/R2CNN_FPN_Tensorflow/blob/master/tools/eval1.py)
    rewrite for angleNet.

2. `sklearn` is a mistake, discard it rightway.

Yongliang Chen
"""

from gpu_IoU.gpu_IoU import gpu_it_IoU
import numpy as np
from iouProb_metric import gen_label_dict, parseLine
import os




def voc_ap(rec, prec, use_07_metric=False):
  """
  average precision calculations
  [precision integrated to recall]
  :param rec: recall
  :param prec: precision
  :param use_07_metric: 2007 metric is 11-recall-point based AP
  :return: average precision
  """
  if use_07_metric:
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
        ap += p / 11.
  else:
    # append sentinel values at both ends
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute precision integration ladder
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # look for recall value changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # sum (\delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap


def angleNet_eval(rboxes, gboxes, iou_th, use_07_metric, mode):
  #  i think i should do some syntax notes for this call...
  rbox_images = rboxes.keys()
  fp = np.zeros(len(rbox_images))
  tp = np.zeros(len(rbox_images))
  box_num = 0

  for i in range(len(rbox_images)):
    rbox_image = rbox_images[i]
    if len(rboxes[rbox_image][0]['bbox']) > 0:

      rbox_lists = np.array(rboxes[rbox_image][0]['bbox'])
      if len(gboxes[rbox_image]) > 0:
#        print gboxes[rbox_image]
        gbox_list = np.array([obj['bbox'] for \
                      obj in gboxes[rbox_image]])
        box_num = box_num + len(gbox_list)
        gbox_list = np.concatenate((gbox_list,\
                np.zeros((np.shape(gbox_list)[0], 1))), axis=1)
        confidence = rbox_lists[:, -1]
        box_index = np.argsort(-confidence)

        rbox_lists = rbox_lists[box_index, :]
        for idx, rbox_list in enumerate(rbox_lists):
          rbox_list = np.reshape(rbox_list, (1,-1))
          if mode == 0:
            ixmin = np.maximum(gbox_list[:, 0], rbox_list[0])
            iymin = np.maximum(gbox_list[:, 1], rbox_list[1])
            ixmax = np.minimum(gbox_list[:, 2], rbox_list[2])
            iymax = np.minimum(gbox_list[:, 3], rbox_list[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih
            # union
            uni = ((rbox_list[2] - rbox_list[0] + 1.) * (rbox_list[3] - rbox_list[1] + 1.) +
                   (gbox_list[:, 2] - gbox_list[:, 0] + 1.) *
                   (gbox_list[:, 3] - gbox_list[:, 1] + 1.) - inters)
            overlaps = inters / uni
          else:
#            overlaps = iou_rotate.iou_rotate_calculate1(np.array([rbox_list[:-1]]),
#                                        gbox_list,
#                                        use_gpu=False)[0]
#            print gbox_list.shape, rbox_list.shape
#            assert 0
#            print rbox_list, gbox_list[0]
            overlaps = gpu_it_IoU(gbox_list[:,:-1],\
                        rbox_list[:,:-1],10, ctx= mx.cpu()).reshape((-1,))
#            print gbox_list.shape, idx, overlaps.max()
#            assert 0

          ovmax = np.max(overlaps)
          jmax = np.argmax(overlaps)
          if ovmax > iou_th:
            if gbox_list[jmax, -1] == 0:
              tp[i] += 1
              gbox_list[jmax, -1] = 1
            else:
              fp[i] += 1
          else:
            fp[i] += 1
      else:
        fp[i] += len(rboxes[rbox_image][0]['bbox'])
    else:
      continue
  rec = np.zeros(len(rbox_images))
  prec = np.zeros(len(rbox_images))
  if box_num == 0:
    for i in range(len(fp)):
      if fp[i] != 0:
        prec[i] = 0
      else:
        prec[i] = 1
  else:
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    rec = tp / box_num
  ap = voc_ap(rec, prec, use_07_metric)

  return rec, prec, ap, box_num


if __name__ == '__main__':
  import mxnet as mx
  size_i = 896/2
  box_dir = '../output/metric-model/AG/%d-Apr-19/boxes'%size_i
 # box_dir = '../output/metric-model/baseline/R2CNN/%d_448_%d/boxes'%(size_i,size_i)
  imgdir='../data/metric/%d'%size_i; labelfile='../data/metric/angleList-%d.txt'%size_i
  for directory in [box_dir, imgdir]:
    assert os.path.isdir(directory), directory

  # construct rbox...
  txt_files = [  x for x in  os.listdir(box_dir) if '.png' in x ]
  assert len(txt_files) > 0
  exname = lambda x: x[len('result-boxes-'):-len('.txt')]
  img_list = [ exname(x) for x in txt_files]
  rboxes = {}
  for f in txt_files:
    txt_path = os.path.join(box_dir,f)
    boxes = np.loadtxt(txt_path , delimiter=',')
    rboxes[exname(f)] = [{'bbox': boxes},]
#    print rbox.keys()

    # construct gbox...
    d = gen_label_dict(labelfile)
    gboxes = {}

    for imgname in img_list:
      s = d[imgname]
      im_info, feat_shape, gdt, img = parseLine(s, imgdir)
      # each box is a obj...
      gboxes[imgname] =[ {'bbox': gdt[i][1:]} \
                        for i in range(len(gdt))]
#  print gboxes.keys()
#  print gboxes['34.png'][0]['bbox'].shape
#  assert 0

  # mix them together...
  rec, prec, ap, box_num = angleNet_eval(rboxes, gboxes, .5, not True,1)
  print rec, prec, ap, box_num

