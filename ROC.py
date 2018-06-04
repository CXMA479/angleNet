from sklearn import metrics
import numpy as np
import os
import matplotlib.pyplot as plt

y= np.random.uniform(0,1,(100,))
l= np.random.randint(0,2,(100,)).astype(int)
#print(y,l)
fpr, tpr, th = metrics.roc_curve(l,y,pos_label=1)
plt.plot(fpr, tpr)
#print(fpr)
#print(fpr[:10])
plt.show()


assert 0


txt_dir = '../data/metric/iou-prob'
outdir = txt_dir
iou_th=.5

model_type_list=['AA_model','AG_model']
file_list = os.listdir(txt_dir)
for model_type in model_type_list:
    label=[]
    score = []
    for txt_file in file_list:
        if model_type not in txt_file:
            continue
        mat = np.loadtxt( os.path.join(txt_dir, txt_file) )
        label +=  list( 1*(mat[:,0]>iou_th) )
        score += list( mat[:,1] )
        fpr,tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
    plt.figure()
    plt.plot(fpr, tpr)
    savename = os.path.join(outdir,model_type+'.txt')
    np.savetxt(savename , np.array([fpr,tpr]).T )
plt.show()

