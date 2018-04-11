import numpy as np
import matplotlib.pyplot as plt


def cp2p( cp,  alpha, rh,  rw):
	"""
            return the four vertices gaven the centerPoint and angle, etc
	"""
	dirH=np.array([np.cos(alpha),np.sin(alpha)])
	dirW=np.array([np.sin(alpha),-np.cos(alpha)])
	return np.array( [ _r+cp for _r in  [rh*dirH+rw*dirW, -rh*dirH+rw*dirW, -rh*dirH-rw*dirW, rh*dirH-rw*dirW]   ]).transpose()


def M(beta):
	return np.array([ [np.cos(beta), -np.sin(beta)],\
		                [np.sin(beta),  np.cos(beta)]])




# will be transfered to cython.

"""
def cp2s(cp,alpha,rh,rw):
	
           return the four edges for the gaven centralPoint and angle, etc
	   format:
			x1 x2 ...
			y1 y2
			x2 x3
			y2 y3
	
	
"""


def has_intersec(bx1,bx2):
	"""
      bx:
           x,y,alphR,rh,rw
	"""

	P1,P2=  [  cp2p( bx[:2], bx[2], bx[3], bx[4]  ) for bx in [bx1, bx2] ]
#	print P1
#	print P2


	# get rotated points to handle whole-embeded case
	P1_on_P2s, P2_on_P2s = [ np.dot( M(- bx2[2]), P)  for P in [P1, P2]   ]

	P1_on_P1s, P2_on_P1s = [ np.dot( M(- bx1[2]), P) for P in [P1, P2]    ]

#	print P1_on_P2s,P2_on_P2s

#	drawAngleBox(P1_on_P2s,'green')
#	drawAngleBox(P2_on_P2s)

	p2s_x_bound=[np.min(P2_on_P2s[0,:]) , np.max(P2_on_P2s[0,:])]
	p2s_y_bound=[np.min(P2_on_P2s[1,:]) , np.max(P2_on_P2s[1,:])]

	p1s_x_bound=[np.min(P1_on_P1s[0,:]) , np.max(P1_on_P1s[0,:])]
	p1s_y_bound=[np.min(P1_on_P1s[1,:]) , np.max(P1_on_P1s[1,:])]

#	print p2s_x_bound,p2s_y_bound
#	p1_idx=1
#	print P1_on_P2s[0,p1_idx] , P1_on_P2s[0,p1_idx],P1_on_P2s[1,p1_idx] ,P1_on_P2s[1,p1_idx]
	for p1_idx in xrange(4):

	# 1st, check the point's bound

#		have not PASSED test yet !

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

			if np.dot(v1,v3) < 0 and np.dot(v2,v4)<0:
#				print "p1_idx:%d, p2_idx:%d"%(p1_idx,p2_idx)
				return True
	return False





def drawAngleBox(vecP,color='red'):
	plt.plot(np.array( [vecP[0,0], vecP[0,1] ]),np.array([vecP[1,0],vecP[1,1]]),color=color,marker='*')
	plt.plot(np.array( [vecP[0,1], vecP[0,2] ]),np.array([vecP[1,1],vecP[1,2]]),color=color,marker='*')
	plt.plot(np.array( [vecP[0,2], vecP[0,3] ]),np.array([vecP[1,2],vecP[1,3]]),color=color,marker='*')
	plt.plot(np.array( [vecP[0,0], vecP[0,-1] ]),np.array([vecP[1,0],vecP[1,-1]]),color=color,marker='*')
	plt.axis('equal')



if __name__ == '__main__':

	R1=np.array([10,5,np.deg2rad(0),8,4])
	R2=np.array([-18,-10,np.deg2rad(0),8,4])


	R1=np.array([10,5,np.deg2rad(30),8,4])
	R2=np.array([0,0,np.deg2rad(60),8,4])

	P1,P2 = [cp2p(R[:2],R[2],R[3],R[4]) for R in [R1, R2]]

#	drawAngleBox(P1,'green')
#	drawAngleBox(P2)
	print has_intersec(R1,R2)
	plt.show()








