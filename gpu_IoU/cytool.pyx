#!python
"""
      need a hand?
      Chen Y.Liang
      Apr 29, 2017
"""
from __future__ import division
import numpy as np

cimport numpy as np
DTYPE=np.float
ctypedef np.float_t DTYPE_t
cimport cython


@cython.boundscheck(False)  #   uncomment it when test ok.
def M(DTYPE_t beta):
#def M(beta):
	return np.array([ [np.cos(beta), -np.sin(beta)],\
	                [np.sin(beta),  np.cos(beta)]])


@cython.boundscheck(False)   #  uncomment it when test ok.
def cp2p(np.ndarray[DTYPE_t, ndim=1] cp, DTYPE_t alpha, DTYPE_t rh, DTYPE_t rw):
#def cp2p( cp,  alpha, rh,  rw):
	"""
            return the four vertices gaven the centerPoint and angle, etc
	"""
	dirH=np.array([np.cos(alpha),np.sin(alpha)])
	dirW=np.array([np.sin(alpha),-np.cos(alpha)])
	return np.array( [ _r+cp for _r in  [rh*dirH+rw*dirW, -rh*dirH+rw*dirW, -rh*dirH-rw*dirW, rh*dirH-rw*dirW]   ]).transpose()


@cython.boundscheck(False)   #  uncomment it when test ok.
def has_intersec(np.ndarray[DTYPE_t, ndim=2] P1, np.ndarray[DTYPE_t, ndim=2] P2, DTYPE_t alpha1, DTYPE_t alpha2):
	"""
      bx:
           x,y,alphR,rh,rw
	"""
	#  already converted.
#	P1,P2=  [  cp2p( np.array( [bx[0], bx[1]]), bx[2], bx[3], bx[4]  ) for bx in [bx1, bx2] ]


	# get rotated points to handle whole-embeded case
	P1_on_P2s, P2_on_P2s = [ np.dot( M(- alpha1), P)  for P in [P1, P2]   ]

	P1_on_P1s, P2_on_P1s = [ np.dot( M(- alpha2), P) for P in [P1, P2]    ]

#	drawAngleBox(P1_on_P2s,'green')
#	drawAngleBox(P2_on_P2s)

	p2s_x_bound=[np.min(P2_on_P2s[0,:]) , np.max(P2_on_P2s[0,:])]
	p2s_y_bound=[np.min(P2_on_P2s[1,:]) , np.max(P2_on_P2s[1,:])]

	p1s_x_bound=[np.min(P1_on_P1s[0,:]) , np.max(P1_on_P1s[0,:])]
	p1s_y_bound=[np.min(P1_on_P1s[1,:]) , np.max(P1_on_P1s[1,:])]

	for p1_idx in xrange(4):

	# 1st, check the point's bound

		if   ( ( P1_on_P2s[0,p1_idx] < p2s_x_bound[1]) and (P1_on_P2s[0,p1_idx] > p2s_x_bound[0]) and \
		       ( P1_on_P2s[1,p1_idx] < p2s_y_bound[1]) and (P1_on_P2s[1,p1_idx] > p2s_y_bound[0] ) ) \
		     or \
		     ( (P2_on_P1s[0,p1_idx] < p1s_x_bound[1]) and  (P2_on_P1s[0,p1_idx] > p1s_x_bound[0]) and \
		       (P2_on_P1s[1,p1_idx] < p1s_y_bound[1]) and  (P2_on_P1s[1,p1_idx] > p1s_y_bound[0]) ):

#			print 'return from embeded points'
			return True


		for p2_idx in xrange(4):
		# so you get k1, k2 e1, e2
			p1_idx_p =(p1_idx+1)%4
			p2_idx_p =(p2_idx+1)%4
#		                 A                      C
			k1 = P1[:, p1_idx ] - P2[:,     p2_idx  ]

#			         B                      D
			k2 = P1[:,p1_idx_p] - P2[:, p2_idx_p]

			e1 = P1[:,     p1_idx]  - P1[:, p1_idx_p]

			e2 = P2[:,     p2_idx]  - P2[:, p2_idx_p]

#			k1e2 = np.dot(k1,e2)
#			k1e1 = np.dot(k1,e1)
#			k2e2 = np.dot(k2,e2)
#			k2e1 = np.dot(k2,e1)
			e1_2 = np.dot(e1,e1)   #np.linalg.norm(e1)**2
			e2_2 = np.dot(e2,e2)   #np.linalg.norm(e2)**2

			v1 = k1 + np.dot(k1,e2)/e2_2 *e2
			v2 =-k1 - np.dot(k1,e1)/e1_2 *e1

			v3 = k2 - np.dot(k2,e2)/e2_2 *e2
			v4 =-k2 + np.dot(k2,e1)/e1_2 *e1
#			print k1
#			print k2
#			print e1
#			print e2

			if np.dot(v1,v3) < 0 and np.dot(v2,v4)<0:   #   so they have a conflict area.
#				print "p1_idx:%d, p2_idx:%d"%(p1_idx,p2_idx)
				return True
	
	return False





@cython.boundscheck(False)  #   uncomment it when test ok.
def iou(np.ndarray[DTYPE_t,ndim=1] anchor, np.ndarray[DTYPE_t,ndim=1] gdt,np.int_t x_num):
#def iou( anchor, gdt, x_num):
	"""
	  A novel numerical approach to estimate IoU :) !
           for single case
                  anchor,gdt :  x,y alphaR, rh,rw

	            |y
	            |  /
		    | /
		    |/alpha
	            ------->x
	"""
	alpha1 = anchor[2]    # radian !
	rh1    = anchor[3]
	rw1    = anchor[4]

	alpha2 = gdt[2]
	rh2    = gdt[3]
	rw2    = gdt[4]

	P1 = cp2p(anchor[0:2],alpha1,rh1,rw1)
	P2 = cp2p(   gdt[0:2],alpha2,gdt[3],gdt[4])

	if not has_intersec(P1,P2,anchor[2], gdt[2]):   # only for fast processing !
		return 0

	P1_mt=np.dot(M(-alpha1),P1)
	P2_mt=np.dot(M(-alpha2),P2)

	p1_tr=np.max(P1_mt,1)
	p1_bl=np.min(P1_mt,1)
	p2_tr=np.max(P2_mt,1)
	p2_bl=np.min(P2_mt,1)

	xlin=( np.linspace(p1_bl[0],p1_tr[0],x_num+2))
	y_num=np.int(np.round(x_num*1./rh1*rw1))         # keep equal space!
	ylin=( np.linspace(p1_bl[1],p1_tr[1],y_num+2)  )

	X,Y=np.meshgrid(xlin[1:x_num+1],ylin[1:y_num+1])  # inner points

	X_bord_1,Y_bord_1 = np.meshgrid(xlin,np.array( (ylin[0],ylin[y_num+1]) ))     # on the border
	X_bord_2,Y_bord_2 = np.meshgrid(np.array( (xlin[0],xlin[x_num+1]) ),ylin[1:y_num+1])

	X_bord =np.concatenate( (X_bord_1.flatten(), X_bord_1.flatten()) )
	Y_bord =np.concatenate( (Y_bord_1.flatten(), Y_bord_1.flatten()) )

	grid_on_P1_space = np.vstack( (X.flatten(),Y.flatten()) )   # not include border
	grid_on_P1_space_bord =np.vstack( (X_bord, Y_bord) )

	grid_on_P2_space      = np.dot( M(alpha1-alpha2),grid_on_P1_space)
	grid_on_P2_space_bord = np.dot( M(alpha1-alpha2),grid_on_P1_space_bord)
#	grid_on_P2_space      = np.dot( M(alpha1-alpha2),grid_on_P1_space)
#	grid_on_P2_space_bord = np.dot( M(alpha1-alpha2),grid_on_P1_space_bord)

	p_in = grid_on_P2_space
	p_bo = grid_on_P2_space_bord

	inner_cnt=np.sum( (p_in[0,:]>p2_bl[0])* (p_in[0,:]< p2_tr[0]) * (p_in[1,:]>p2_bl[1])* (p_in[1,:]<p2_tr[1]) )
	bord_cnt =np.sum( (p_bo[0,:]>p2_bl[0])* (p_bo[0,:]< p2_tr[0]) * (p_bo[1,:]>p2_bl[1])* (p_bo[1,:]<p2_tr[1]) )


	# calculate the IoU
	weight_cnt=inner_cnt*2 + bord_cnt
	weight_total_cnt = grid_on_P1_space.size/2 *2 + grid_on_P1_space_bord.size/2
	S_of_R1 = weight_cnt*1./weight_total_cnt
	S_R1= 4*rw1*rh1
	S_R2= 4*rw2*rh2

	return S_of_R1 *S_R1/ (S_R1+S_R2-S_of_R1 *S_R1 +np.exp(-4))








@cython.boundscheck(False)   #  uncomment it when test ok.
def it_IoU(np.ndarray[DTYPE_t,ndim=2] anchor, np.ndarray[DTYPE_t,ndim=2] gdt,np.int_t x_num):
#def it_IoU(anchor,  gdt,x_num):
	"""
	anchor:  num x  5 # 5:       x,y,alphaR,rh,rw
                                            0 1    2   3  4

	gdt   :  n  x 5         # 5: x , y , alphaR , rh , rw
	                                0    1   2      3      4   

    return: iou_matrix  
                       : num x n
	"""

	anchor_num = anchor.shape[0]
	gdt_num    = gdt.shape[0]

	iou_matrix=np.empty( (anchor_num, gdt_num) ,dtype=np.float32)
	for i in np.arange(anchor_num):
		for j in np.arange(gdt_num):
			iou_matrix[i,j]=iou(anchor[i],gdt[j],x_num)
	return iou_matrix



"""
if __name__ == '__main__': # do some test
#import matplotlib.pyplot
	R1=np.array([10,5,np.deg2rad(0),8,4])
	R2=np.array([-18,-10,np.deg2rad(0),8,4])
	print 'IoU:',iou(R1,R2,20)

	R1=np.array([10,5,np.deg2rad(0),8,4])
	R2=np.array([18,1,np.deg2rad(0),8,4])
	print 'IoU:',iou(R1,R2,20)
"""









