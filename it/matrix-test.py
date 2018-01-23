import numpy as np

matName='matrix'

#f=open('matrix.txt','wb+')
img1=np.random.uniform(0,1,(68*70*6,20))
img2=np.random.uniform(0,5,(4,5))

#np.savez_compressed(matName+'-compress',x=img1)
np.savez(matName+'-noCompress',x=img1,y=img2)

ld=np.load(matName+'-noCompress.npz')
ld['x']



np.save(matName+'-bin',img1)                  #   USE THIS ONE!
f.write(str(img1))
f.close()


bin_img = np.load(matName+'-bin.npy')
(bin_img-img1).sum()


#f=open('matrix.txt','rb')
#mats=np.load(f)
#s=f.readlines()
#x=mats['x']
#print s
