# coding=utf-8


"""

            one alternative way to sovle this problem
                   count the sub-rectangles falling into the other

            Matrix transfering is needed.

                Chen Y.Liang
"""

import numpy as np
import matplotlib.pyplot as plt

rh1=7
rw1=5
P1c=np.array([0,0])
alpha1=np.deg2rad(40)


x_num=10
#y_num=10



rh2=4
rw2=2
P2c=np.array([5,6])
alpha2=np.deg2rad(-40)


def MTran(beta):
	'''
		beta is radians!
	'''
	M=np.matrix([[np.cos(beta), -np.sin(beta)],\
		     [np.sin(beta),  np.cos(beta)]])
	return M


def drawAngleBox(vecP,color='red'):
	plt.plot(np.array( [vecP[0,0], vecP[0,1] ]),np.array([vecP[1,0],vecP[1,1]]),color=color,marker='*')
	plt.plot(np.array( [vecP[0,1], vecP[0,2] ]),np.array([vecP[1,1],vecP[1,2]]),color=color,marker='*')
	plt.plot(np.array( [vecP[0,2], vecP[0,3] ]),np.array([vecP[1,2],vecP[1,3]]),color=color,marker='*')
	plt.plot(np.array( [vecP[0,0], vecP[0,-1] ]),np.array([vecP[1,0],vecP[1,-1]]),color=color,marker='*')
	plt.axis('equal')


def cp2p(cp,alpha,rh,rw):
	"""
            return the four vertices gaven the centerPoint and angle, etc
	"""
	dirH=np.array([np.cos(alpha),np.sin(alpha)])
	dirW=np.array([np.sin(alpha),-np.cos(alpha)])
	return np.array( [ _r+cp for _r in  [rh*dirH+rw*dirW, -rh*dirH+rw*dirW, -rh*dirH-rw*dirW, rh*dirH-rw*dirW]   ]).transpose()





P1=cp2p(P1c,alpha1,rh1,rw1)
P2=cp2p(P2c,alpha2,rh2,rw2)


P1_mt=np.dot(MTran(-alpha1),P1)
P2_mt=np.dot(MTran(-alpha2),P2)

#print np.dot(P1[:,0]-P1[:,1],P1[:,1]-P1[:,2])
#print P1



#print P1
#print P3

p1_tr=np.array(np.max(P1_mt,1).reshape(2,))[0]
p1_bl=np.array(np.min(P1_mt,1).reshape(2,))[0]

p2_tr=np.array(np.max(P2_mt,1).reshape(2,))[0]
p2_bl=np.array(np.min(P2_mt,1).reshape(2,))[0]


#print p1_bl
#print p1_tr

xlin=( np.linspace(p1_bl[0],p1_tr[0],x_num+2))#[1:-1]

y_num=np.int(np.round(x_num*1./rh1*rw1))
ylin=( np.linspace(p1_bl[1],p1_tr[1],y_num+2)  )#[1:-1]




#plt.plot(xlin,ylin,marker='.',ls='None')

X,Y=np.meshgrid(xlin,ylin)

X_bord=np.concatenate( ( X[0,:],X[1:-1,0],X[1:-1,-1],X[-1,:] ))
Y_bord=np.concatenate( (  Y[0,:],Y[1:-1,0],Y[1:-1,-1],Y[-1,:]))

X=X[1:-1,1:-1]
Y=Y[1:-1,1:-1]
grid_on_P1_space=np.array([ [X.reshape(-1,1)],[Y.reshape(-1,1)]] ).reshape(2,-1)

grid_on_P1_space_bord=np.array([ [X_bord.reshape(-1,1)],[Y_bord.reshape(-1,1)]] ).reshape(2,-1)

#plt.plot(grid_on_P1_space[0,:],grid_on_P1_space[1,:],marker='+',ls='None')



grid_on_P2_space=np.array( np.dot( MTran(alpha1-alpha2),grid_on_P1_space))
p=grid_on_P2_space
p_whole=np.dot( MTran(alpha2), p)
plt.plot(p_whole[0,:],p_whole[1,:],marker='x',color='blue',ls='None')
intersectionP=p[  :,(p[0,:]>p2_bl[0])*(p[0,:]<p2_tr[0])  * (p[1,:]>p2_bl[1])* (p[1,:]<p2_tr[1]) ] # attached to P2_mt
interP_on_origin=np.dot( MTran(alpha2), intersectionP)
plt.plot(interP_on_origin[0,:],interP_on_origin[1,:],marker='x',ls='None',color='red')

grid_on_P2_space_bord=np.array( np.dot( MTran(alpha1-alpha2),grid_on_P1_space_bord))
p_bord=grid_on_P2_space_bord
p_whole_bord=np.dot( MTran(alpha2), p_bord)
plt.plot(p_whole_bord[0,:],p_whole_bord[1,:],marker='.',color='blue',ls='None')
intersectionP_bord=p_bord[  :,(p_bord[0,:]>=p2_bl[0])*(p_bord[0,:]<p2_tr[0])  * (p_bord[1,:]>p2_bl[1])* (p_bord[1,:]<p2_tr[1]) ] # attached to P2_mt
interP_on_origin_bord=np.dot( MTran(alpha2), intersectionP_bord)
plt.plot(interP_on_origin_bord[0,:],interP_on_origin_bord[1,:],marker='.',ls='None',color='red')


S= 1.*(interP_on_origin[0,:].size*2+interP_on_origin_bord[0,:].size)/(p_whole[0,:].size*2+p_whole_bord[0,:].size)

legend_str='('+str(interP_on_origin[0,:].size)+'x2+'+str(interP_on_origin_bord[0,:].size)+')/('+str(p_whole[0,:].size)+'x2+'+str(p_whole_bord[0,:].size)+')='+str(S)

plt.title(legend_str)
drawAngleBox(P1,'red')
drawAngleBox(P2,'blue')
#drawAngleBox(P1_mt,'green')


plt.figure()
drawAngleBox(P1,'red')
drawAngleBox(P1_mt,'green')


plt.figure()
drawAngleBox(P1,'red')
drawAngleBox(P1_mt,'green')
plt.plot(grid_on_P1_space[0,:],grid_on_P1_space[1,:],marker='+',ls='None')
plt.plot(grid_on_P1_space_bord[0,:],grid_on_P1_space_bord[1,:],marker='+',ls='None')



plt.figure()
drawAngleBox(P2,'blue')
drawAngleBox(P2_mt,'green')
plt.plot(grid_on_P2_space[0,:],grid_on_P2_space[1,:],marker='+',ls='None')
plt.plot(grid_on_P2_space_bord[0,:],grid_on_P2_space_bord[1,:],marker='+',ls='None')



plt.figure()
drawAngleBox(P2,'blue')
drawAngleBox(P1,'red')

plt.show()



