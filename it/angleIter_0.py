"""
         I need a raw read iter
             Chen Y. Liang
             April 29, 2017
"""


import matplotlib.pyplot as plt
import mxnet as mx
import os,sys
import numpy as np
import logging
import time

import cv2

sys.path.insert(0,'../')   # tool.py
sys.path.insert(0,'../cython')  # cytool.so
import tool
#import cytool

from config import config #logger
try:
	os.system('make -C ../cython') # newest one
	from cytool import it_IoU  # cython implementation
except:
	os.system('rm ../cython/cytool.so')
	os.system('make -C ../cython')
	from cytool import it_IoU

nd= mx.nd
npr=np.random

def post_iou(iou_matrix,anchor,gdt):
	"""
		iou_matrix: n x m

		anchor    : n x 5        x,y,alphaR,rh,rw
		gdt       : m x 6  label,

                   U must return the final version:

		  return target_label, target_bbx, target_weight
	"""
	anchor_num = anchor.shape[0]

	target_bbx    = anchor.copy()

	target_label  = np.empty( (anchor_num,) ,dtype=np.float32)

	#  1: objective, 0: background, -1: not care
	target_label.fill(-1)

	target_weight = np.zeros( (anchor_num,5),dtype=np.float32)
	if gdt.size > 0:

		# 1st. IoU>THRESHOLD_HIGH  ,label
		pred_indx, gdt_indx=np.where(iou_matrix>config.train.THRESHOLD_OBJECT)
#		logging.debug('[size] target_bbx gdt:'+str(target_bbx.shape)+' '+str(gdt.shape) )
		target_bbx[pred_indx, :] = gdt[gdt_indx, 1:]
#		target_weight[pred_indx][:] =1
		target_label[pred_indx]     =1     # only need to state wheather it is object or background

		# 2nd. each gdt must have its own agent , label
		pred_indx=iou_matrix.argmax(0)
		target_bbx[pred_indx,:] = gdt[:,1:]
#		target_weight[pred_indx][:] =1
		target_label[pred_indx]     =1

		# 3ed. remove IoU < THRESHOLD_LOW ,  label
		pred_indx=np.where(iou_matrix<config.train.THRESHOLD_BACKGD)[0]
#		target_weight[pred_indx][:] =0
		target_label[pred_indx]     =0
	else:
		target_label[:]=0

	#   subsample...  keeping balance is important
	max_fg  = int(config.train.rpn_fg_frac * config.train.rpn_batch_size)
	fg_indx = np.where(target_label==1)[0]  # how many objectives are there
	logging.debug('ojectives num:%d'%len(fg_indx))
	if len(fg_indx) > max_fg:
		disable_indx = npr.choice(fg_indx,size(len(fg_indx)-max_fg),replace=False)
		target_label[disable_indx] = -1

	# same story for backgd
	max_bg = config.train.rpn_batch_size - np.sum(target_label == 1)
	bg_indx = np.where(target_label == 0)[0]
	logging.debug('background num:%d'%len(bg_indx))
	if len(bg_indx) > max_bg:
		disable_indx = npr.choice( bg_indx, size=(len(bg_indx) - max_bg),replace=False)
		target_label[disable_indx] = -1

	target_weight[target_label ==1, :]= np.array(config.train.rpn_bbx_weight)


	#  now noliner Transfer...
	if gdt.size > 0:
		target_bbx = tool.bbx_transform(anchor,target_bbx)

	return target_label, target_bbx, target_weight



class angleIter(mx.io.DataIter):
	"""
              labelFile
			file format:
			name.type;	l	x,y	alpha	rh	rw;	l...
	      imgPath

	      epoch

	      it

	      dataName
			'img','label','gdt'

	"""
	def __init__(self,symGp=None):
		"""
                        labelName: used for anchor-wise regression
			symGp : mx.symbol.Group([])
			stride: pixels(in raw img) between to neighbor anchors' centrs
			sideLength: \sqrt{Aera} , 
			                          Aera is fixed for one sideLength wrt. angleD
		"""
#		assert  ( symGp is None and config.DEBUG ) or (symGp is not None and not config.DEBUG)
		if symGp is None:
			assert config.debug.debug




#		self.threshold_high=config.it.train.THRESHOLD_HIGH
#		self.threshold_low =config.it.train.THRESHOLD_LOW


		super(angleIter,self).__init__()
		self.x_num  =config.train.iou_x_num
		self.angleD = config.ANCHOR_angleD
		self.stride = config.ANCHOR_stride
		self.HoW    = config.ANCHOR_HoW
		self.sideLength = config.ANCHOR_sideLength
		self.epoch=config.it.epoch
		self.it=config.it.iteration
		self.shuffle=config.it.shuffle

		self.dataName=config.it.dataName
		self.labelName=config.it.labelName

		assert os.path.isfile(config.it.labelFile)
		self.fileName = config.it.labelFile
		assert os.path.isdir(config.it.imgPath)
		self.imgPath = config.it.imgPath


		if config.it.debug.feat_shape is not None:
			self.feat_shape = config.it.debug.feat_shape

		self.fs=open(self.fileName).readlines()
		self.imgNum=len(self.fs)
#		assert self.imgNum == len(os.listdir(self.imgPath))
		self.cur_it=0
		self.cur_epoch=0
		self.lineIdx=0   #   line index of the img
		self.lineStr=''
		self.imgName=''
		self.lg=''
		self.provide_label=[]
		self.provide_data=[]
		# get the symobl ofthe feature map
		self.feat_sym=symGp.get_internals()[config.net.rpn_conv_name+'_output']  if config.it.debug.feat_shape is None else None
			
		self.target_label=[]
		self.target_gdtbox=[]
		self.target_weight=[]
		self.gdt=[]
		self.im_info=[]
		self.img=0
		self.type_num = len(self.HoW)*len(self.angleD)*len(self.sideLength)
		ioudir_name = str(config.ANCHOR_angleD)+str(config.ANCHOR_HoW) + str(config.ANCHOR_sideLength)+\
				str(config.ANCHOR_stride)
		
		self.ioudir = os.path.join(config.it.anchordir_path,ioudir_name)
		self.hasiou_flag = os.path.isdir(self.ioudir)
		if not self.hasiou_flag: # make one
			os.mkdir(self.ioudir)


		if config.it.debug.debug:
			self.anchor=0


	def reset(self):
		self.cur_it=0


	def checkAndSet(self):
		if self.cur_it >= self.it:
			self.cur_epoch +=1
			raise StopIteration
		self.cur_it +=1

	def genIdx(self):
		if self.shuffle:
			self.lineIdx = np.random.randint(0,self.imgNum)
		else:
			self.lineIdx +=1
			if self.lineIdx >= self.imgNum:
				self.lineIdx =0


	def parseLine(self):
		"""
                       all convert to matrix, including img (read it and convert to matrix)
		           and get the feature map's shape
		"""
		self.lineStr=self.fs[self.lineIdx]
		self.lg = self.lineStr.rsplit(';\t')
		self.imgName = self.lg.pop(0)
		self.img=mx.image.imdecode(open(os.path.join(self.imgPath,self.imgName),'rb').read())


		self.lgSize=len(self.lg)   # as Batch Channel
		self.imgHWC=self.img.shape
		if config.debug.debug:
			logging.debug('image shape:'+str(self.imgHWC))
		H,W,C=self.imgHWC
		if self.feat_sym is not None:
			_, self.feat_shape, _=self.feat_sym.infer_shape(**{self.dataName[0]:(1,C,H,W)})  # it only needs img
			self.feat_shape = self.feat_shape[0]

		self.provide_data=[(self.dataName[0],(1,C,H,W)), \
				   (self.dataName[1],(self.lgSize,1,1,6)),\
				   (self.dataName[2],(2,))]    # H,W

		#     label, bbx, weight
		feat_shape = self.feat_shape
		logging.debug('feat_shape:%s'%(str(feat_shape)))
		self.provide_label=[(self.labelName[0],(feat_shape[-2]*feat_shape[-1]*self.type_num,1)),\
				    (self.labelName[1],(feat_shape[-2]*feat_shape[-1]*self.type_num,5)),\
				    (self.labelName[2],(feat_shape[-2]*feat_shape[-1]*self.type_num,5))]


		self.gdt=np.zeros((self.lgSize,6))
		self.im_info=np.array([H,W])

		self.lg.append(self.lg.pop(-1)[:-1])  # get rid of '\n'
		#   go for 'label', 'gdt'
		for i,obj in enumerate(self.lg):
			"""
                                 now obj:  label	x,y	alpha(deg)	rh	rw
			"""
			label,xy,alphaD,rh,rw=obj.split('\t')
			label,alphaD,rh,rw = [ np.float(_) for _ in (label,alphaD,rh,rw)]
			x,y=[np.float(_) for _ in xy.rsplit(',') ]

			alpha=np.deg2rad(alphaD)  # deg 2 rad

			self.gdt[i][:] = np.array([label,x,y,alpha,rh,rw])



	def checkout_ioufile(self):
		return os.path.isdir(self.ioudir)


	def next(self):  # one img per call
		self.checkAndSet()
		self.genIdx()
		self.parseLine()
		#   havnt ended yet! longway to go
		#   assign gdt for each anchor, but 1st, u need generate anchors
		
		#   an_gdtbox IS NOT gdtbox FOR NOW !
		anchor = tool.genAnchor(self.im_info,self.angleD, self.HoW, self.sideLength,\
										self.feat_shape,self.stride) # anchor : N x 5
#		logging.debug('anchor size:'+str(anchor.shape))
#		logging.debug('anchor data:'+str(anchor))

		if config.it.debug.debug:
			self.anchor=anchor

		#   IoU ?
		start=time.time() if config.debug.debug else None
		ioufile = os.path.join(self.ioutdir,self.imgName[:-4])+'_'+str(self.imgHWC[:2])+'.npy'   # encode the additional id

#		if self.hasiou_flag and not os.path.isfile(ioufile): # need to notify the owner
#			logging.info('ioufile does not exsists: '+ioufile)

		if not self.hasiou_flag or not os.path.isfile(ioufile): # calculate and save it
			iou_matrix = it_IoU(anchor,     self.gdt[:,1:],   \
								self.x_num)    #  BE CAREFUL ! [:,0] IS FOR LABEL !
			np.save(ioufile,iou_maxtrix)
		else:  # just load it
			iou_matrix = np.load(ioufile)



		if config.it.debug.debug:
			logging.debug('[IoU] time elapse:'+str(time.time()-start))
			logging.debug('[IoU] maximum:%f'%np.max(iou_matrix))
#			logging.debug('iou matrix:'+str(iou_matrix))
#			logging.debug('begin to calculate IoU...')
#			logging.debug('anchor:'+str(anchor))
#			logging.debug('gdt:'+str(self.gdt))

			#  display the img with anchors and gdt onit !
			img=self.img.asnumpy()
			img=tool.draw_angleBox(img,anchor,(0,250,0))
			img=tool.draw_angleBox(img,self.gdt[:,1:],(0,0,250))
			plt.imshow(img)
			plt.show()


		self.target_label,self.target_bbx, self.target_weight =post_iou(iou_matrix,anchor,self.gdt)




		self.img=nd.transpose(self.img,axes=(2,0,1))
		self.img=nd.expand_dims(self.img,axis=0)


		return mx.io.DataBatch( [self.img,self.gdt,self.im_info],[self.target_label,self.target_bbx, self.target_weight])


